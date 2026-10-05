//! reactor — `mio` / `io_uring` masquerade backed by the hashtrinity Table.
//!
//! ## The collapse: readiness and completion are one demux, keyed by token
//!
//! `mio` (epoll/kqueue) and `io_uring` look like two subsystems. They are one:
//! a **token-keyed demux over a single [`Table`]**, differing only by *when* the token comes back.
//!
//! ```s
//! registry = Table<usize, Interest>   -- certify once: register = insert_with (Interest merged by OR)
//! wait     = the one syscall cut       -- epoll_wait  (readiness)  |  io_uring_enter (completion)
//! events   = Table<usize, Readiness>  -- the demux: which tokens came back, and how
//! ```
//!
//! * **mio / epoll — readiness.** A [`Token`] is a `usize`, so it *is* its own coordinate (no hash,
//!   one probe — see the crate root). The kernel stores the token as `epoll_event.data` and hands it
//!   back per ready fd, so the *kernel* does the fd→token association; the userspace [`Table`] is the
//!   token→interest/state map every reactor needs — the `HashMap<Token, Conn>` mio forces on you,
//!   made a coordinate read (O(1), HashDOS-immune).
//! * **io_uring — completion.** A submission carries `user_data: u64` — the *same* token. The
//!   completion queue hands it back: "this token's op is done." The CQE demux is the *same* token
//!   read; the buffer the op owned is a slot in a `Table<usize, Buf>` pool, returned at the
//!   coordinate the CQE names. Streaming law — certify the buffer once, read it back at its
//!   coordinate — not a "buffer manager tracking hardware lifecycle."
//!
//! So there is **one** primitive; readiness vs completion is the [`Driver`], not a fork in the data.
//! The io_uring "drop incompatibility" (a future dropped mid-flight while the kernel writes its
//! buffer) dissolves the same way the Table's [`insert_with`](crate::Table::insert_with) merge
//! dissolves double-free: ownership is moved into the ring's pool before the danger and returned by
//! the completion — the future never owns the in-flight buffer, so dropping it frees nothing.
//!
//! ## The one cut (Axiom XI): pure Rust, no `.c`, no mmap
//!
//! The kernel readiness primitive isn't in `std`, so the [`Driver`] seam wraps a syscall. The real
//! [`EpollDriver`] (feature `reactor-epoll`) issues its **own** `epoll_create1`/`epoll_ctl`/
//! `epoll_wait` through one inline-`asm!` `syscall` (linux/x86_64) — the Rust twin of
//! `proof/native/sanguine_syscall.c (unpublished)`: no dep, no libc, no C, no `build.rs`. epoll fills a plain
//! userspace buffer — **no mmap** (that is only io_uring's optional ring-elision, and even it has a
//! NO_MMAP mode). All `unsafe` is encapsulated in the driver's private `sys` module behind safe
//! wrappers; the rest of the reactor is safe. The `reactor` core is the safe *model* + [`MockDriver`]:
//! zero deps, zero `unsafe`, zero syscalls, so it tests the registration→demux path standalone.
//!
//! ## Feature parity
//!
//! | mio / io_uring                     | hashtrinity `reactor`                    | status |
//! |---|---|---|
//! | `mio::Token(usize)`                | [`Token`] (usize, its own coordinate)    | ✓ |
//! | `mio::Interest` (READABLE/WRITABLE)| [`Interest`] (OR-merged bitset)          | ✓ |
//! | `mio::event::Event` readiness      | [`Readiness`] bitset                     | ✓ |
//! | `mio::Events` (event buffer)       | [`Events`] = `Table<usize, Readiness>`   | ✓ |
//! | `mio::Poll` register/reregister    | [`Reactor`] register/reregister          | ✓ |
//! | `mio::Registry::deregister`        | [`Reactor::deregister`]                  | ✓ |
//! | `mio::Poll::poll`                  | [`Reactor::poll`] (via [`Driver`])       | ✓ |
//! | epoll backend                      | [`EpollDriver`] (own syscall cut, feat reactor-epoll) | ✓ |
//! | io_uring `user_data` demux         | same token read, [`Readiness::DONE`]     | ~ (model; native driver is next rung) |
//! | io_uring registered fixed buffers  | `Table<usize, Buf>` pool                 | − (NB: needs core `Table::remove` for reuse) |

use crate::Table;
use std::io;
use std::os::fd::BorrowedFd;

/// A registration handle — a `usize` that is its own coordinate. Matches `mio::Token`. On io_uring
/// this is the submission's `user_data`; the completion hands it back to name the finished op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(pub usize);

/// What a source is registered to wait for (readiness) — an OR-merged bitset, so re-registering a
/// token *adds* interest. Matches `mio::Interest`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Interest(u8);

/// What actually came back from a wait — the same bitset shape as [`Interest`]. For epoll this is the
/// readiness (can-read/can-write); for io_uring, `DONE` marks a completed op. Matches
/// `mio::event::Event`'s readiness bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Readiness(u8);

impl Interest {
    pub const READABLE: Interest = Interest(0b01);
    pub const WRITABLE: Interest = Interest(0b10);
    /// Union — `reregister` merges old and new interest with this.
    #[inline]
    pub fn add(self, other: Interest) -> Interest {
        Interest(self.0 | other.0)
    }
    #[inline]
    pub fn is_readable(self) -> bool {
        self.0 & Self::READABLE.0 != 0
    }
    #[inline]
    pub fn is_writable(self) -> bool {
        self.0 & Self::WRITABLE.0 != 0
    }
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    #[inline]
    pub fn bits(self) -> u8 {
        self.0
    }
}

impl Readiness {
    pub const READABLE: Readiness = Readiness(0b0001);
    pub const WRITABLE: Readiness = Readiness(0b0010);
    /// io_uring completion: the op named by this token finished.
    pub const DONE: Readiness = Readiness(0b0100);
    /// A multishot operation (`submit_recv_multishot`) has stopped generating completions —
    /// the kernel's `IORING_CQE_F_MORE` came back unset (buffer-pool exhaustion or an error) —
    /// and needs `submit_recv_multishot` called again to keep receiving. Rides the same
    /// OR-merged bitset as READABLE/WRITABLE/DONE rather than a separate accessor, so a caller
    /// checking one `Events` read sees it alongside the rest.
    pub const ENDED: Readiness = Readiness(0b1000);
    #[inline]
    pub fn from_bits(bits: u8) -> Readiness {
        Readiness(bits)
    }
    #[inline]
    pub fn add(self, other: Readiness) -> Readiness {
        Readiness(self.0 | other.0)
    }
    #[inline]
    pub fn is_readable(self) -> bool {
        self.0 & Self::READABLE.0 != 0
    }
    #[inline]
    pub fn is_writable(self) -> bool {
        self.0 & Self::WRITABLE.0 != 0
    }
    #[inline]
    pub fn is_done(self) -> bool {
        self.0 & Self::DONE.0 != 0
    }
    #[inline]
    pub fn is_ended(self) -> bool {
        self.0 & Self::ENDED.0 != 0
    }
    #[inline]
    pub fn bits(self) -> u8 {
        self.0
    }
}

/// The demux out of one wait: `Table<usize, Readiness>`, token → what's ready. `mio::Events`
/// (readiness) and the io_uring CQE batch (completion) are *this* — the same container as the
/// registry, one face over. A driver merges each returned event by OR, so a token both readable and
/// writable in one wait lands as one entry.
pub struct Events {
    ready: Table<usize, Readiness>,
    /// A completed read/write operation's result (byte count, or a negative errno) — separate
    /// from `ready` because a completion result isn't a readiness bit; a poll completion never
    /// populates this, a read/write completion always does. `IoUringDriver`-specific in practice
    /// (no other driver submits read/write operations), but lives on the shared `Events` type
    /// because it's demuxed by the same token, through the same one `wait()` call.
    results: Table<usize, i32>,
}

impl Events {
    pub fn new() -> Self {
        Events {
            ready: Table::new(),
            results: Table::new(),
        }
    }

    /// Record that `token` came back ready/done for `r` (OR-merged with any prior in this batch).
    pub fn record(&mut self, token: Token, r: Readiness) {
        self.ready.insert_with(token.0, r, |new, old| old.add(new));
    }

    /// Record a completed read/write operation's result for `token` (byte count, or a negative
    /// errno — the raw `cqe.res`, same convention as a syscall's raw return value).
    pub fn record_result(&mut self, token: Token, res: i32) {
        self.results.insert_with(token.0, res, |new, _old| new);
    }

    /// The readiness for a token in this batch, or empty.
    pub fn get(&self, token: Token) -> Readiness {
        self.ready.get(&token.0).copied().unwrap_or_default()
    }

    /// A completed read/write operation's result for a token in this batch, if any.
    pub fn result(&self, token: Token) -> Option<i32> {
        self.results.get(&token.0).copied()
    }

    /// Iterate the ready tokens. `mio`'s `for event in &events`.
    pub fn iter(&self) -> impl Iterator<Item = (Token, Readiness)> + '_ {
        self.ready.iter().map(|(k, v)| (Token(*k), *v))
    }

    pub fn len(&self) -> usize {
        self.ready.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }

    /// Drop all events, keeping the table, before the next wait. `mio::Events::clear`.
    pub fn clear(&mut self) {
        self.ready = Table::new();
        self.results = Table::new();
    }
}

impl Default for Events {
    fn default() -> Self {
        Self::new()
    }
}

/// The one hardware cut, behind a seam. A driver owns the registration lifecycle (so a real backend
/// can `epoll_ctl`) and the wait. Readiness vs completion is *which driver*; the demux above does not
/// change. The fd is a [`BorrowedFd`] — the reactor never owns or closes it, and never needs `unsafe`
/// to hold a raw fd. [`MockDriver`] ignores the fd; [`EpollDriver`] passes it to `epoll_ctl`.
pub trait Driver {
    /// Register `fd` under `token` for `interest` (fresh token).
    fn register(&mut self, token: Token, fd: BorrowedFd<'_>, interest: Interest) -> io::Result<()>;
    /// Replace a token's interest (mio semantics: reregister sets, does not union at the kernel).
    fn reregister(
        &mut self,
        token: Token,
        fd: BorrowedFd<'_>,
        interest: Interest,
    ) -> io::Result<()>;
    /// Stop waiting on `token`/`fd`.
    fn deregister(&mut self, token: Token, fd: BorrowedFd<'_>) -> io::Result<()>;
    /// Wait up to `timeout_ms` (`None` = block) and fill `events` with ready/completed tokens.
    /// Returns the number of events produced.
    fn wait(&mut self, events: &mut Events, timeout_ms: Option<u32>) -> io::Result<usize>;
}

/// The masquerade entry point — `mio::Poll`. Holds one [`Driver`] and forwards the registration
/// lifecycle to it.
pub struct Reactor<D: Driver> {
    driver: D,
}

impl<D: Driver> Reactor<D> {
    pub fn new(driver: D) -> Self {
        Reactor { driver }
    }

    pub fn register(
        &mut self,
        token: Token,
        fd: BorrowedFd<'_>,
        interest: Interest,
    ) -> io::Result<()> {
        self.driver.register(token, fd, interest)
    }
    pub fn reregister(
        &mut self,
        token: Token,
        fd: BorrowedFd<'_>,
        interest: Interest,
    ) -> io::Result<()> {
        self.driver.reregister(token, fd, interest)
    }
    pub fn deregister(&mut self, token: Token, fd: BorrowedFd<'_>) -> io::Result<()> {
        self.driver.deregister(token, fd)
    }

    /// Wait once, filling `events`. `mio::Poll::poll`. The demux is read at each token's coordinate;
    /// there is no scan of a readiness list.
    pub fn poll(&mut self, events: &mut Events, timeout_ms: Option<u32>) -> io::Result<usize> {
        events.clear();
        self.driver.wait(events, timeout_ms)
    }

    /// Direct driver access (used in tests to script a [`MockDriver`]).
    pub fn driver(&mut self) -> &mut D {
        &mut self.driver
    }
}

/// A syscall-free driver for exercising and testing the model: readiness is *scripted* rather than
/// read from the kernel. It keeps its own token→interest [`Table`] (the registration set), so a
/// scripted readiness is only reported for a registered token with matching interest — real epoll
/// semantics, with zero `unsafe` and zero syscalls. The fd is ignored.
#[derive(Default)]
pub struct MockDriver {
    interests: Table<usize, Interest>,
    scripted: Vec<(Token, Readiness)>,
}

impl MockDriver {
    pub fn new() -> Self {
        MockDriver {
            interests: Table::new(),
            scripted: Vec::new(),
        }
    }
    /// Queue a readiness the next `wait` will report for `token` (if registered with matching interest).
    pub fn ready(&mut self, token: Token, r: Readiness) {
        self.scripted.push((token, r));
    }
    fn interest(&self, token: Token) -> Interest {
        self.interests.get(&token.0).copied().unwrap_or_default()
    }
}

impl Driver for MockDriver {
    fn register(
        &mut self,
        token: Token,
        _fd: BorrowedFd<'_>,
        interest: Interest,
    ) -> io::Result<()> {
        self.interests
            .insert_with(token.0, interest, |new, _old| new);
        Ok(())
    }
    fn reregister(
        &mut self,
        token: Token,
        _fd: BorrowedFd<'_>,
        interest: Interest,
    ) -> io::Result<()> {
        self.interests
            .insert_with(token.0, interest, |new, _old| new);
        Ok(())
    }
    fn deregister(&mut self, token: Token, _fd: BorrowedFd<'_>) -> io::Result<()> {
        self.interests
            .insert_with(token.0, Interest::default(), |new, _old| new);
        Ok(())
    }
    fn wait(&mut self, events: &mut Events, _timeout_ms: Option<u32>) -> io::Result<usize> {
        let mut n = 0;
        for (token, r) in std::mem::take(&mut self.scripted) {
            let interest = self.interest(token);
            if interest.is_empty() {
                continue; // epoll never reports a fd you didn't register
            }
            let masked =
                Readiness::from_bits(r.bits() & (interest.bits() | Readiness::DONE.bits()));
            if masked.bits() != 0 {
                events.record(token, masked);
                n += 1;
            }
        }
        Ok(n)
    }
}

// ── The real epoll driver — our own syscall cut (linux/x86_64), no deps, no C, no mmap. ──────────

#[cfg(feature = "ht-reactor-epoll")]
mod epoll_driver {
    use super::{Driver, Events, Interest, Readiness, Token};
    use std::io;
    use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

    /// The one cut into silicon (Axiom XI): the raw `syscall` ABI, args as data — the Rust twin of
    /// `proof/native/sanguine_syscall.c (unpublished)`. Every `unsafe` in the reactor lives in this module; the
    /// wrappers it exposes are safe, and everything above them is safe. Numbers are the Linux x86_64
    /// table (data, cf. `src/kernel/Syscall.s`); no libc, no C shim, no `build.rs`.
    mod sys {
        use std::io;
        use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};

        const SYS_EPOLL_WAIT: usize = 232;
        const SYS_EPOLL_CTL: usize = 233;
        const SYS_EPOLL_CREATE1: usize = 291;
        #[cfg(test)]
        const SYS_READ: usize = 0;
        #[cfg(test)]
        const SYS_WRITE: usize = 1;
        #[cfg(test)]
        const SYS_PIPE2: usize = 293;
        // (close is the kernel `close(2)` = 3, but `OwnedFd`'s Drop issues it — no wrapper needed here)

        const CLOEXEC: usize = 0x8_0000; // O_CLOEXEC == EPOLL_CLOEXEC
        const EINTR: isize = -4;

        pub const EPOLL_CTL_ADD: i32 = 1;
        pub const EPOLL_CTL_DEL: i32 = 2;
        pub const EPOLL_CTL_MOD: i32 = 3;
        pub const EPOLLIN: u32 = 0x001;
        pub const EPOLLOUT: u32 = 0x004;
        pub const EPOLLERR: u32 = 0x008;
        pub const EPOLLHUP: u32 = 0x010;

        /// The single instruction cut. x86_64 convention: `nr` in `rax`, args in `rdi/rsi/rdx/r10`,
        /// result in `rax`; the `syscall` insn clobbers `rcx`/`r11`. Four args covers every syscall
        /// the reactor uses (`epoll_ctl`/`epoll_wait` are the widest). This is the only `unsafe`.
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        #[inline]
        unsafe fn syscall4(nr: usize, a0: usize, a1: usize, a2: usize, a3: usize) -> isize {
            let ret: isize;
            core::arch::asm!(
                "syscall",
                inlateout("rax") nr => ret,
                in("rdi") a0, in("rsi") a1, in("rdx") a2, in("r10") a3,
                lateout("rcx") _, lateout("r11") _,
                options(nostack),
            );
            ret
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        compile_error!(
            "the reactor-epoll Driver's raw syscall cut is implemented for linux/x86_64 only; \
             extend `syscall4` with the target's ABI (aarch64: `svc #0`, nr in x8, args x0..x3)"
        );

        #[inline]
        fn check(r: isize) -> io::Result<usize> {
            if r < 0 {
                Err(io::Error::from_raw_os_error(-r as i32))
            } else {
                Ok(r as usize)
            }
        }

        /// The kernel's `epoll_event`, packed on x86_64: `{ u32 events; u64 data; }` = 12 bytes.
        /// `data` carries the token; `epoll_wait` returns it verbatim per ready fd.
        #[repr(C, packed)]
        #[derive(Clone, Copy)]
        pub struct EpollEvent {
            pub events: u32,
            pub data: u64,
        }

        pub fn epoll_create1() -> io::Result<OwnedFd> {
            // SAFETY: epoll_create1(EPOLL_CLOEXEC) — returns a fresh fd or -errno.
            let fd = check(unsafe { syscall4(SYS_EPOLL_CREATE1, CLOEXEC, 0, 0, 0) })? as RawFd;
            // SAFETY: `fd` is a fresh, owned descriptor the kernel just handed us.
            Ok(unsafe { OwnedFd::from_raw_fd(fd) })
        }

        pub fn epoll_ctl(
            epfd: BorrowedFd,
            op: i32,
            fd: BorrowedFd,
            ev: &EpollEvent,
        ) -> io::Result<()> {
            // SAFETY: epoll_ctl(epfd, op, fd, ev) — `ev` is a valid pointer for the duration.
            check(unsafe {
                syscall4(
                    SYS_EPOLL_CTL,
                    epfd.as_raw_fd() as usize,
                    op as usize,
                    fd.as_raw_fd() as usize,
                    ev as *const EpollEvent as usize,
                )
            })
            .map(|_| ())
        }

        pub fn epoll_wait(
            epfd: BorrowedFd,
            buf: &mut [EpollEvent],
            timeout: i32,
        ) -> io::Result<usize> {
            loop {
                // SAFETY: epoll_wait(epfd, ptr, len, timeout) — the kernel writes up to `len` events
                // into the exclusively-borrowed `buf`.
                let r = unsafe {
                    syscall4(
                        SYS_EPOLL_WAIT,
                        epfd.as_raw_fd() as usize,
                        buf.as_mut_ptr() as usize,
                        buf.len(),
                        timeout as usize,
                    )
                };
                if r == EINTR {
                    continue; // interrupted before any event — retry
                }
                return check(r);
            }
        }

        /// A real pipe + write, so the epoll test needs no external crate. `pipe2(&fds, O_CLOEXEC)`.
        #[cfg(test)]
        pub fn pipe2_cloexec() -> io::Result<(OwnedFd, OwnedFd)> {
            let mut fds = [0i32; 2];
            // SAFETY: pipe2(fds_ptr, O_CLOEXEC) — fills fds[0]=read, fds[1]=write on success.
            check(unsafe { syscall4(SYS_PIPE2, fds.as_mut_ptr() as usize, CLOEXEC, 0, 0) })?;
            // SAFETY: two fresh owned fds from the kernel.
            Ok(unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) })
        }
        #[cfg(test)]
        pub fn write(fd: BorrowedFd, bytes: &[u8]) -> io::Result<usize> {
            // SAFETY: write(fd, ptr, len) from a valid, readable slice.
            check(unsafe {
                syscall4(
                    SYS_WRITE,
                    fd.as_raw_fd() as usize,
                    bytes.as_ptr() as usize,
                    bytes.len(),
                    0,
                )
            })
        }
        #[cfg(test)]
        pub fn read(fd: BorrowedFd, buf: &mut [u8]) -> io::Result<usize> {
            // SAFETY: read(fd, ptr, len) into a valid, exclusively-borrowed slice.
            check(unsafe {
                syscall4(
                    SYS_READ,
                    fd.as_raw_fd() as usize,
                    buf.as_mut_ptr() as usize,
                    buf.len(),
                    0,
                )
            })
        }
    }

    // Test-only syscall helpers, shared with the `runtime` module's tests (a real pipe + read/write).
    #[cfg(test)]
    pub(crate) use sys::{pipe2_cloexec, read, write};

    /// epoll behind the [`Driver`] seam. Holds the epoll `OwnedFd` and a reusable event buffer. The
    /// token lives in `epoll_event.data` and comes back per ready fd, so the kernel does the fd→token
    /// association; interest maps to `EPOLLIN`/`EPOLLOUT`, readiness maps back. No mmap — `epoll_wait`
    /// fills the plain `Vec` through the ordinary syscall return.
    pub struct EpollDriver {
        epfd: OwnedFd,
        buf: Vec<sys::EpollEvent>,
    }

    impl EpollDriver {
        /// Create the epoll instance (`epoll_create1(EPOLL_CLOEXEC)`), sized for `capacity` events
        /// per wait.
        pub fn new(capacity: usize) -> io::Result<Self> {
            Ok(EpollDriver {
                epfd: sys::epoll_create1()?,
                buf: vec![sys::EpollEvent { events: 0, data: 0 }; capacity.max(1)],
            })
        }
    }

    fn to_flags(interest: Interest) -> u32 {
        let mut f = 0;
        if interest.is_readable() {
            f |= sys::EPOLLIN;
        }
        if interest.is_writable() {
            f |= sys::EPOLLOUT;
        }
        f
    }

    fn from_flags(flags: u32) -> Readiness {
        let mut r = Readiness::from_bits(0);
        // HUP/ERR surface as readable so a one-shot read observes EOF/error, matching mio.
        if flags & (sys::EPOLLIN | sys::EPOLLHUP | sys::EPOLLERR) != 0 {
            r = r.add(Readiness::READABLE);
        }
        if flags & sys::EPOLLOUT != 0 {
            r = r.add(Readiness::WRITABLE);
        }
        r
    }

    impl Driver for EpollDriver {
        fn register(
            &mut self,
            token: Token,
            fd: BorrowedFd<'_>,
            interest: Interest,
        ) -> io::Result<()> {
            let ev = sys::EpollEvent {
                events: to_flags(interest),
                data: token.0 as u64,
            };
            sys::epoll_ctl(self.epfd.as_fd(), sys::EPOLL_CTL_ADD, fd, &ev)
        }
        fn reregister(
            &mut self,
            token: Token,
            fd: BorrowedFd<'_>,
            interest: Interest,
        ) -> io::Result<()> {
            let ev = sys::EpollEvent {
                events: to_flags(interest),
                data: token.0 as u64,
            };
            sys::epoll_ctl(self.epfd.as_fd(), sys::EPOLL_CTL_MOD, fd, &ev)
        }
        fn deregister(&mut self, _token: Token, fd: BorrowedFd<'_>) -> io::Result<()> {
            let ev = sys::EpollEvent { events: 0, data: 0 };
            sys::epoll_ctl(self.epfd.as_fd(), sys::EPOLL_CTL_DEL, fd, &ev)
        }
        fn wait(&mut self, events: &mut Events, timeout_ms: Option<u32>) -> io::Result<usize> {
            let timeout = match timeout_ms {
                Some(ms) => ms.min(i32::MAX as u32) as i32,
                None => -1, // block
            };
            let n = sys::epoll_wait(self.epfd.as_fd(), &mut self.buf, timeout)?;
            for ev in self.buf[..n].iter().copied() {
                events.record(Token(ev.data as usize), from_flags(ev.events));
            }
            Ok(n)
        }
    }

    // ── The real boundary: epoll over a real pipe, our syscalls, no C, no mmap. ──
    #[cfg(test)]
    mod tests {
        use super::super::{Events, Interest, Reactor, Readiness, Token};
        use super::{sys, EpollDriver};
        use std::os::fd::AsFd;

        #[test]
        fn epoll_reports_a_readable_pipe() {
            let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
            let mut reactor = Reactor::new(EpollDriver::new(16).expect("epoll_create1"));
            reactor
                .register(Token(1), rd.as_fd(), Interest::READABLE)
                .unwrap();

            // nothing written yet → a 0ms wait sees no readiness
            let mut events = Events::new();
            assert_eq!(
                reactor.poll(&mut events, Some(0)).unwrap(),
                0,
                "empty pipe not readable"
            );

            // write one byte → the read end is readable, and epoll hands back token 1
            sys::write(wr.as_fd(), b"x").expect("write");
            let n = reactor.poll(&mut events, Some(1000)).unwrap();
            assert_eq!(n, 1, "epoll reported the readable pipe");
            assert!(
                events.get(Token(1)).is_readable(),
                "token 1 demuxed to readable"
            );
            let _ = Readiness::DONE; // (completion face shares this demux; exercised in the model tests)

            reactor.deregister(Token(1), rd.as_fd()).unwrap();
        }
    }
}

#[cfg(feature = "ht-reactor-epoll")]
pub use epoll_driver::EpollDriver;

// The `runtime` module's tests need a real pipe + read/write; re-export the driver's test helpers.
#[cfg(all(test, feature = "ht-reactor-epoll"))]
pub(crate) use epoll_driver::{pipe2_cloexec, read, write};

// ── The real io_uring driver — same demux, a completion-based backend. ───────────────────────────

// PRIVATE module, re-exported below at the path it already had: no new public path exists, so
// `hashtrinity::reactor::IoUringDriver` is unchanged and nothing downstream needed opening.
#[cfg(feature = "ht-reactor-iouring")]
mod iouring_driver;

#[cfg(feature = "ht-reactor-iouring")]
pub use iouring_driver::IoUringDriver;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::os::fd::AsFd;

    // A real fd for the mock's ignored `fd` parameter — /dev/null, no `unsafe`.
    fn devnull() -> File {
        File::open("/dev/null").expect("open /dev/null")
    }

    #[test]
    fn token_is_its_own_coordinate() {
        let mut r = Reactor::new(MockDriver::new());
        let f = devnull();
        r.register(Token(7), f.as_fd(), Interest::READABLE).unwrap();
        assert_eq!(r.driver().interest(Token(7)), Interest::READABLE);
        assert!(r.driver().interest(Token(8)).is_empty());
    }

    #[test]
    fn reregister_replaces_interest() {
        let mut r = Reactor::new(MockDriver::new());
        let f = devnull();
        r.register(Token(1), f.as_fd(), Interest::READABLE).unwrap();
        r.reregister(Token(1), f.as_fd(), Interest::WRITABLE)
            .unwrap();
        assert_eq!(r.driver().interest(Token(1)), Interest::WRITABLE);
    }

    #[test]
    fn deregister_tombstones_to_empty() {
        let mut r = Reactor::new(MockDriver::new());
        let f = devnull();
        r.register(Token(3), f.as_fd(), Interest::READABLE).unwrap();
        r.deregister(Token(3), f.as_fd()).unwrap();
        assert!(r.driver().interest(Token(3)).is_empty());
    }

    #[test]
    fn poll_demuxes_only_registered_tokens() {
        let mut r = Reactor::new(MockDriver::new());
        let f = devnull();
        r.register(Token(10), f.as_fd(), Interest::READABLE)
            .unwrap();
        r.register(Token(20), f.as_fd(), Interest::WRITABLE)
            .unwrap();
        r.driver().ready(Token(10), Readiness::READABLE);
        r.driver().ready(Token(20), Readiness::WRITABLE);
        r.driver().ready(Token(99), Readiness::READABLE); // never registered — dropped
        let mut events = Events::new();
        let n = r.poll(&mut events, Some(0)).unwrap();
        assert_eq!(n, 2);
        assert!(events.get(Token(10)).is_readable());
        assert!(events.get(Token(20)).is_writable());
        assert!(events.get(Token(99)).bits() == 0);
    }

    #[test]
    fn interest_masks_readiness() {
        let mut r = Reactor::new(MockDriver::new());
        let f = devnull();
        r.register(Token(5), f.as_fd(), Interest::READABLE).unwrap();
        r.driver().ready(Token(5), Readiness::WRITABLE);
        let mut events = Events::new();
        r.poll(&mut events, Some(0)).unwrap();
        assert!(
            events.get(Token(5)).bits() == 0,
            "readiness outside interest is dropped"
        );
    }

    #[test]
    fn completion_demux_reuses_the_same_table() {
        let mut r = Reactor::new(MockDriver::new());
        let f = devnull();
        r.register(Token(42), f.as_fd(), Interest::READABLE)
            .unwrap();
        r.driver().ready(Token(42), Readiness::DONE);
        let mut events = Events::new();
        r.poll(&mut events, Some(0)).unwrap();
        assert!(
            events.get(Token(42)).is_done(),
            "completion is the same token-keyed demux"
        );
    }
}
