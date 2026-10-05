//! The io_uring backend — the completion face of the one token-keyed demux described in the
//! parent module. Was `mod iouring_driver { .. }` inline in `reactor.rs`; the module PATH is   CITATION-EXEMPT: the pre-split monolith; it is `src/reactor/mod.rs` now
//! unchanged (`crate::reactor::iouring_driver`), so `use super::{..}` below still names the
//! reactor model and `IoUringDriver` is still re-exported as `reactor::IoUringDriver`.

use super::{Driver, Events, Interest, Readiness, Token};
use crate::Table;
use std::collections::VecDeque;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

/// The one cut into silicon (Axiom XI), io_uring's version: `io_uring_setup`/`io_uring_enter`
/// through the raw `syscall` ABI (the same instruction epoll's driver uses, a second
/// independent binding of it — Axiom XI is "the syscall is the cut," not "there is exactly
/// one syscall"), plus `mmap`/`munmap` for the kernel-shared SQ/CQ rings (the one place this
/// driver needs memory-mapped shared state instead of a plain buffer — io_uring's actual
/// mechanism, not something to route around). Struct layouts and constants below are
/// transcribed directly from `/usr/include/linux/io_uring.h` (byte-verified against that
/// header, not derived from memory) to rule out a hand-derived-offset bug in the one place
/// that would be silent and catastrophic — wrong struct layout means garbage submissions, not
/// a compile error.
mod sys {
    use std::io;
    use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};

    const SYS_IO_URING_SETUP: usize = 425;
    const SYS_IO_URING_ENTER: usize = 426;
    const SYS_IO_URING_REGISTER: usize = 427;
    const SYS_MMAP: usize = 9;
    const SYS_MUNMAP: usize = 11;
    const EINTR: isize = -4;
    #[cfg(test)]
    const SYS_READ: usize = 0;
    #[cfg(test)]
    const SYS_WRITE: usize = 1;
    #[cfg(test)]
    const SYS_PIPE2: usize = 293;
    #[cfg(test)]
    const SYS_SOCKETPAIR: usize = 53;
    #[cfg(test)]
    const CLOEXEC: usize = 0x8_0000;
    #[cfg(test)]
    const AF_UNIX: usize = 1;
    #[cfg(test)]
    const SOCK_STREAM: usize = 1;

    pub const PROT_READ: usize = 0x1;
    pub const PROT_WRITE: usize = 0x2;
    pub const MAP_SHARED: usize = 0x01;
    pub const MAP_PRIVATE: usize = 0x02;
    pub const MAP_ANONYMOUS: usize = 0x20;
    pub const MAP_POPULATE: usize = 0x008000;

    pub const IORING_OFF_SQ_RING: i64 = 0;
    pub const IORING_OFF_SQES: i64 = 0x10000000;

    pub const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;
    pub const IORING_ENTER_GETEVENTS: u32 = 1 << 0;
    pub const IORING_ENTER_SQ_WAKEUP: u32 = 1 << 1;

    /// A kernel-side thread polls the SQ ring on its own — submitting new work is then a
    /// plain memory write (publish the tail), no syscall, as long as the thread hasn't gone
    /// to sleep (see `IORING_SQ_NEED_WAKEUP` below). Needs no elevated capability for the
    /// basic (unpinned) case; `IORING_SETUP_SQ_AFF` (CPU-pinning the thread) is the part that
    /// wants `CAP_SYS_NICE`, and this driver doesn't set it.
    pub const IORING_SETUP_SQPOLL: u32 = 1 << 1;
    /// Set by the kernel in the SQ ring's `flags` field when the polling thread has gone to
    /// sleep (idle for `sq_thread_idle` ms with nothing to do) — userspace must notice this
    /// and pass `IORING_ENTER_SQ_WAKEUP` on its next `io_uring_enter` to bring it back.
    pub const IORING_SQ_NEED_WAKEUP: u32 = 1 << 0;

    pub const IORING_OP_POLL_ADD: u8 = 6;
    pub const IORING_OP_POLL_REMOVE: u8 = 7;
    pub const IORING_OP_READ: u8 = 22;
    pub const IORING_OP_WRITE: u8 = 23;
    pub const IORING_OP_RECV: u8 = 27;

    pub const IORING_POLL_ADD_MULTI: u32 = 1 << 0;

    pub const POLLIN: u32 = 0x0001;
    pub const POLLOUT: u32 = 0x0004;
    pub const POLLERR: u32 = 0x0008;
    pub const POLLHUP: u32 = 0x0010;

    /// `sqe->ioprio`'s recv/send-specific flag: keep generating a CQE per arrival from ONE
    /// submitted SQE — the B/U certification this driver's `submit_recv_multishot` exists
    /// for (see that method's doc): certify the recv ONCE, read every subsequent arrival at
    /// its own coordinate (the buffer id the CQE names), never resubmit.
    pub const IORING_RECV_MULTISHOT: u16 = 1 << 1;
    /// `sqe->flags` bit: pick the buffer from `buf_index_or_group`'s registered pool instead
    /// of the caller supplying one via `addr`/`len` — required alongside `IORING_OP_RECV` +
    /// `IORING_RECV_MULTISHOT`; the kernel can't know which buffer to write into across many
    /// not-yet-arrived completions otherwise.
    pub const IOSQE_BUFFER_SELECT: u8 = 1 << 5;
    /// `cqe->flags` bit: the upper 16 bits of `flags` carry the buffer id the kernel picked
    /// and filled for this completion.
    pub const IORING_CQE_F_BUFFER: u32 = 1 << 0;
    /// `cqe->flags` bit: unset means this multishot has stopped generating completions
    /// (buffer-pool exhaustion or an error) — surfaced to callers as `Readiness::ENDED`.
    pub const IORING_CQE_F_MORE: u32 = 1 << 1;
    /// `cqe->flags` bit: set alongside a terminal (`!F_MORE`) completion to mean "the socket
    /// still has more data waiting — this wasn't EOF, the multishot just ran out of buffers
    /// for THIS attempt." Verified live to matter: treating every `res == 0` terminal
    /// completion as EOF silently truncated real transfers under concurrency (data still
    /// sitting in the kernel's receive buffer got reported as a closed connection) — this bit
    /// is exactly what distinguishes that case from a genuine peer close.
    pub const IORING_CQE_F_SOCK_NONEMPTY: u32 = 1 << 2;

    /// `io_uring_register` opcode: register a provided-buffer ring (`IoUringBufReg`) for a
    /// buffer-group id — the pool `submit_recv_multishot` draws from.
    pub const IORING_REGISTER_PBUF_RING: u32 = 22;
    /// The inverse — required before a `bgid` can be registered again (re-registering an
    /// already-registered `bgid` on the same ring is `EINVAL`/`EEXIST`, verified live). A
    /// caller that reuses `bgid` values (e.g. `fd as u16`, where the kernel legitimately
    /// recycles fd numbers once a socket closes) must unregister the old pool first.
    pub const IORING_UNREGISTER_PBUF_RING: u32 = 23;

    /// Six-argument form — `io_uring_enter`/`mmap` need more than epoll's widest call. Same
    /// instruction, same calling convention, as `epoll_driver::sys::syscall4`.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[inline]
    unsafe fn syscall6(
        nr: usize,
        a0: usize,
        a1: usize,
        a2: usize,
        a3: usize,
        a4: usize,
        a5: usize,
    ) -> isize {
        let ret: isize;
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") nr => ret,
                in("rdi") a0, in("rsi") a1, in("rdx") a2, in("r10") a3, in("r8") a4, in("r9") a5,
                lateout("rcx") _, lateout("r11") _,
                options(nostack),
            );
        }
        ret
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    compile_error!(
        "the reactor-iouring Driver's raw syscall cut is implemented for linux/x86_64 only"
    );

    #[inline]
    fn check(r: isize) -> io::Result<isize> {
        if r < 0 {
            Err(io::Error::from_raw_os_error(-r as i32))
        } else {
            Ok(r)
        }
    }

    // ── Kernel struct layouts, transcribed from linux/io_uring.h (byte-exact, #[repr(C)]). ──

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoSqringOffsets {
        pub head: u32,
        pub tail: u32,
        pub ring_mask: u32,
        pub ring_entries: u32,
        pub flags: u32,
        pub dropped: u32,
        pub array: u32,
        pub resv1: u32,
        pub user_addr: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoCqringOffsets {
        pub head: u32,
        pub tail: u32,
        pub ring_mask: u32,
        pub ring_entries: u32,
        pub overflow: u32,
        pub cqes: u32,
        pub flags: u32,
        pub resv1: u32,
        pub user_addr: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoUringParams {
        pub sq_entries: u32,
        pub cq_entries: u32,
        pub flags: u32,
        pub sq_thread_cpu: u32,
        pub sq_thread_idle: u32,
        pub features: u32,
        pub wq_fd: u32,
        pub resv: [u32; 3],
        pub sq_off: IoSqringOffsets,
        pub cq_off: IoCqringOffsets,
    }

    /// The 64-byte submission queue entry. Only the fields this driver's two operations
    /// (`IORING_OP_POLL_ADD`, `IORING_OP_POLL_REMOVE`) actually use are named; the rest are
    /// explicit padding at the byte offsets the kernel's unions occupy, so the struct's total
    /// size and every named field's offset match `struct io_uring_sqe` exactly. `#[repr(C)]`
    /// with these exact primitive widths reproduces the layout without Rust's less-ergonomic
    /// C-union syntax — verified offset-by-offset against the header (see module doc).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoUringSqe {
        pub opcode: u8,
        pub flags: u8,
        pub ioprio: u16,
        pub fd: i32,
        pub off_addr2: u64, // off / addr2 / (cmd_op,pad1) — unused by POLL_ADD/POLL_REMOVE
        pub addr: u64,      // POLL_REMOVE's target: the token (user_data) being cancelled
        pub len: u32,       // ALSO the POLL_ADD command-flags field (IORING_POLL_ADD_MULTI)
        pub poll32_events: u32, // the POLLIN/POLLOUT mask for POLL_ADD
        pub user_data: u64,
        pub buf_index_or_group: u16,
        pub personality: u16,
        pub splice_fd_in: u32,
        pub addr3: u64,
        pub pad2: u64,
    }

    /// The 16-byte completion queue entry (no `IORING_SETUP_CQE32`, so exactly this size).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoUringCqe {
        pub user_data: u64,
        pub res: i32,
        pub flags: u32,
    }

    /// One provided-buffer descriptor — 16 bytes, `struct io_uring_buf`. `resv` is unused by
    /// the buffer's own addr/len/bid semantics; slot 0's `resv` doubles as the ring's `tail`
    /// counter (the kernel's own union layout, `struct io_uring_buf_ring` — see
    /// `BufRing::recycle`), not a separate field this driver has to model.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoUringBufDesc {
        pub addr: u64,
        pub len: u32,
        pub bid: u16,
        pub resv: u16,
    }

    /// Argument to `IORING_REGISTER_PBUF_RING` — 40 bytes, `struct io_uring_buf_reg`.
    /// `flags` left at 0 (no `IOU_PBUF_RING_MMAP`): this driver supplies its own
    /// already-allocated, already-aligned `Vec<IoUringBufDesc>` memory directly via
    /// `ring_addr` rather than asking the kernel to `mmap` a fresh region for it — one fewer
    /// mmap call, and `Vec<T>`'s allocator already guarantees `align_of::<T>()` (8 bytes,
    /// enough for `IoUringBufDesc`'s `u64` field) and a stable address as long as the `Vec`
    /// is never resized (same reasoning `submit_read`'s doc gives for its own buffer pool).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoUringBufReg {
        pub ring_addr: u64,
        pub ring_entries: u32,
        pub bgid: u16,
        pub flags: u16,
        pub min_left: u32,
        pub resv: [u32; 5],
    }

    pub fn io_uring_setup(entries: u32, params: &mut IoUringParams) -> io::Result<OwnedFd> {
        // SAFETY: io_uring_setup(entries, &mut params) — the kernel reads `params.flags` (left
        // zeroed: no SQPOLL, no fixed files, defaults throughout) and writes ring sizing and
        // the SQ/CQ offset tables back into the same struct.
        let fd = check(unsafe {
            syscall6(
                SYS_IO_URING_SETUP,
                entries as usize,
                params as *mut IoUringParams as usize,
                0,
                0,
                0,
                0,
            )
        })? as RawFd;
        // SAFETY: `fd` is a fresh, owned descriptor the kernel just handed us.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    pub fn io_uring_enter(
        ring_fd: BorrowedFd,
        to_submit: u32,
        min_complete: u32,
        flags: u32,
    ) -> io::Result<usize> {
        loop {
            // SAFETY: io_uring_enter(fd, to_submit, min_complete, flags, NULL, 0) — no signal
            // mask (we never need to atomically unblock a signal across the wait).
            let r = unsafe {
                syscall6(
                    SYS_IO_URING_ENTER,
                    ring_fd.as_raw_fd() as usize,
                    to_submit as usize,
                    min_complete as usize,
                    flags as usize,
                    0,
                    0,
                )
            };
            if r == EINTR {
                continue; // interrupted before completing the wait — retry, matching epoll_wait
            }
            return check(r).map(|v| v as usize);
        }
    }

    pub fn io_uring_register(
        ring_fd: BorrowedFd,
        opcode: u32,
        arg: *const u8,
        nr_args: u32,
    ) -> io::Result<usize> {
        // SAFETY: io_uring_register(fd, opcode, arg, nr_args) — `arg` points at a value
        // valid for the call's duration (an `IoUringBufReg` for `IORING_REGISTER_PBUF_RING`).
        check(unsafe {
            syscall6(
                SYS_IO_URING_REGISTER,
                ring_fd.as_raw_fd() as usize,
                opcode as usize,
                arg as usize,
                nr_args as usize,
                0,
                0,
            )
        })
        .map(|v| v as usize)
    }

    /// Returns the mapped region's start address — kernel-shared memory (the SQ/CQ rings or
    /// the SQE array), never freed by Rust's allocator, unmapped explicitly via `munmap` in
    /// the driver's `Drop`.
    pub fn mmap(len: usize, prot: usize, ring_fd: BorrowedFd, offset: i64) -> io::Result<*mut u8> {
        // SAFETY: mmap(NULL, len, prot, MAP_SHARED|MAP_POPULATE, ring_fd, offset) — the kernel
        // picks the address (first arg 0), maps `len` bytes of the ring/SQE region at the
        // io_uring-defined `offset` (IORING_OFF_SQ_RING/SQES), shared so writes through this
        // pointer are visible to the kernel side of the ring.
        let r = unsafe {
            syscall6(
                SYS_MMAP,
                0,
                len,
                prot,
                MAP_SHARED | MAP_POPULATE,
                ring_fd.as_raw_fd() as usize,
                offset as usize,
            )
        };
        check(r).map(|p| p as usize as *mut u8)
    }

    /// Anonymous, private, page-aligned memory not tied to any fd or io_uring offset — what
    /// `IORING_REGISTER_PBUF_RING` actually needs for a self-supplied (non-`IOU_PBUF_RING_MMAP`)
    /// `ring_addr`: the kernel's own registration path requires page alignment for the pin it
    /// takes on that memory, which a plain heap `Vec<T>` allocation does not guarantee (glibc
    /// aligns small allocations far below the page — this was the exact EINVAL a `Vec`-backed
    /// attempt hit, verified live rather than assumed from documentation). `mmap` always
    /// returns a page-aligned address, so this is the one-line fix, not a workaround.
    pub fn mmap_anon(len: usize) -> io::Result<*mut u8> {
        let r = unsafe {
            syscall6(
                SYS_MMAP,
                0,
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                usize::MAX,
                0,
            )
        };
        check(r).map(|p| p as usize as *mut u8)
    }

    pub fn munmap(ptr: *mut u8, len: usize) {
        // SAFETY: munmap(ptr, len) on a region this driver mmap'd with exactly this length.
        // Best-effort on Drop — nothing meaningful to do with a failure here.
        unsafe {
            syscall6(SYS_MUNMAP, ptr as usize, len, 0, 0, 0, 0);
        }
    }

    // Test-only: a real pipe + write/read, so the io_uring tests need no external crate —
    // independent of `epoll_driver`'s copy so this feature has no cross-feature test dependency.
    #[cfg(test)]
    pub fn pipe2_cloexec() -> io::Result<(OwnedFd, OwnedFd)> {
        let mut fds = [0i32; 2];
        check(unsafe { syscall6(SYS_PIPE2, fds.as_mut_ptr() as usize, CLOEXEC, 0, 0, 0, 0) })?;
        Ok(unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) })
    }
    /// `IORING_OP_RECV` is `recv(2)` semantics — needs a socket, not a pipe. A unix
    /// `socketpair` exercises it with no network/port involved.
    #[cfg(test)]
    pub fn socketpair() -> io::Result<(OwnedFd, OwnedFd)> {
        let mut fds = [0i32; 2];
        check(unsafe {
            syscall6(
                SYS_SOCKETPAIR,
                AF_UNIX,
                SOCK_STREAM,
                0,
                fds.as_mut_ptr() as usize,
                0,
                0,
            )
        })?;
        Ok(unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) })
    }
    #[cfg(test)]
    pub fn write(fd: BorrowedFd, bytes: &[u8]) -> io::Result<usize> {
        check(unsafe {
            syscall6(
                SYS_WRITE,
                fd.as_raw_fd() as usize,
                bytes.as_ptr() as usize,
                bytes.len(),
                0,
                0,
                0,
            )
        })
        .map(|v| v as usize)
    }
    #[cfg(test)]
    pub fn read(fd: BorrowedFd, buf: &mut [u8]) -> io::Result<usize> {
        check(unsafe {
            syscall6(
                SYS_READ,
                fd.as_raw_fd() as usize,
                buf.as_mut_ptr() as usize,
                buf.len(),
                0,
                0,
                0,
            )
        })
        .map(|v| v as usize)
    }
}

use sys::{IoCqringOffsets, IoSqringOffsets, IoUringCqe, IoUringParams, IoUringSqe};

/// One mmap'd region this driver owns and must `munmap` on drop.
struct Mapping {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `Mapping` has no interior mutability of its own — every read/write into the mmap'd
// region happens through `IoUringDriver`'s own `&mut self` methods, so ownership (Send) and
// shared-reference safety (Sync) reduce to whatever the OWNING driver's caller already
// guarantees (a single mutex around the whole driver, same shape `std::fs::File` — also a
// raw-fd wrapper — already gets automatically). Without this, ANY struct containing a
// `Reactor<IoUringDriver>` is structurally `!Send`/`!Sync` even behind a mutex, since Rust's
// auto-trait inference sees the raw pointer field regardless of what wraps it.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

/// A provided-buffer pool for multishot recv — the B/U certification `submit_recv_multishot`
/// reads against forever after `register_recv_ring` runs once. `submit_read`/`submit_write`
/// already apply the streaming law (Axiom V) at the granularity of ONE operation: certify a
/// buffer, read the result back at its token's coordinate, never re-scan. A `BufRing` applies
/// the SAME law one level up — certify a POOL of buffers ONCE per connection (or per driver),
/// then every subsequent arrival is a coordinate read (the CQE names which buffer id got the
/// data) with NO resubmission, closing the fixed per-operation cost `submit_read` still pays
/// once per transfer. `mem` (the descriptor ring) must be `mmap`'d, not a plain heap `Vec`:
/// `IORING_REGISTER_PBUF_RING`'s own pin on `ring_addr` requires page alignment, which a
/// heap allocation doesn't guarantee — verified live (a `Vec`-backed first attempt hit
/// `EINVAL`), not assumed. `data` (the actual receive bytes) has no such requirement — a
/// plain `Vec` is fine there, same reasoning `bufs` above already relies on.
struct BufRing {
    /// `entries` descriptor slots, page-aligned (`sys::mmap_anon`). Slot 0's `resv` field
    /// doubles as the ring's kernel-visible `tail` counter (the kernel's own overlapping
    /// union layout — see `recycle`).
    mem: Mapping,
    /// `entries * buf_size` bytes backing every descriptor's `addr`.
    data: Vec<u8>,
    mask: u16,
    buf_size: u32,
    /// Our own count of buffers ever made available to the kernel (wrapping, masked only
    /// when indexing `mem`) — the same monotonic-counter shape as `sq_local_tail` above.
    local_tail: u16,
}

impl BufRing {
    #[inline]
    unsafe fn desc_slot(&self, idx: u16) -> *mut sys::IoUringBufDesc {
        unsafe { (self.mem.ptr as *mut sys::IoUringBufDesc).add(idx as usize) }
    }

    fn buf_data(&self, buf_id: u16, n: usize) -> &[u8] {
        let off = buf_id as usize * self.buf_size as usize;
        &self.data[off..off + n.min(self.buf_size as usize)]
    }

    /// Make `buf_id` available to the kernel again at the next tail slot, then publish the
    /// new tail — the streaming-law recycle: the SAME certified pool serves the next
    /// arrival, no reallocation, no re-registration.
    fn recycle(&mut self, buf_id: u16) {
        let slot = self.local_tail & self.mask;
        // SAFETY: `data`'s allocation is stable (never resized after `register_recv_ring`),
        // so this address stays valid for the pool's whole lifetime; `slot` is within
        // `[0, entries)` by the mask, and this write is published (Release, below) before
        // the kernel can pick this slot for a new completion.
        let addr = unsafe {
            self.data
                .as_ptr()
                .add(buf_id as usize * self.buf_size as usize)
        } as u64;
        unsafe {
            *self.desc_slot(slot) = sys::IoUringBufDesc {
                addr,
                len: self.buf_size,
                bid: buf_id,
                resv: 0,
            }
        };
        self.local_tail = self.local_tail.wrapping_add(1);
        // SAFETY: byte offset 14 within slot 0 is the kernel's overlaid `tail` field
        // (`struct io_uring_buf_ring`'s union — see the module doc's byte-verification
        // note). Release store publishes the descriptor write above; the kernel Acquire-
        // loads this same field before picking a buffer for the next multishot completion.
        unsafe {
            let tail_ptr = self.mem.ptr.add(14) as *const AtomicU16;
            (*tail_ptr).store(self.local_tail, Ordering::Release);
        }
    }
}

impl Drop for BufRing {
    fn drop(&mut self) {
        sys::munmap(self.mem.ptr, self.mem.len);
    }
}

/// A reserved `user_data` value marking "this completion is the driver's own bookkeeping (a
/// `POLL_REMOVE`'s acknowledgement), not a real readiness event" — `wait()` skips any CQE
/// carrying it. `Token(usize::MAX)` is therefore not usable as a real registration token under
/// this driver specifically (labeled, narrow, Axiom XIII) — no other driver has this
/// reservation.
const REMOVE_SENTINEL: u64 = u64::MAX;

/// io_uring behind the [`Driver`] seam. Registers each token as a **multishot**
/// `IORING_OP_POLL_ADD` (the kernel keeps generating a completion every time the fd's interest
/// is satisfied, with no resubmission needed — level-triggered, matching epoll's default and
/// this trait's documented reregister-replaces semantics), keyed by `user_data = token.0`, the
/// same demux [`Readiness::DONE`] was built for.
///
/// `register`/`reregister`/`deregister` only write the SQE locally — nothing crosses into the
/// kernel until `wait()`, the one `io_uring_enter` call that both publishes every submission
/// queued since the last wait and reaps every completion at once. That's the actual lever over
/// epoll: many connections' registration changes AND their readiness collapse into one
/// syscall's round trip, not one epoll_ctl per change plus one epoll_wait per check. It does
/// NOT remove the read()/write() syscall a caller still pays per I/O operation once a token is
/// ready. That floor is closed by `submit_read`/`submit_write` below — `IORING_OP_READ`/
/// `WRITE` submitted and batched through the exact same `wait()` call, so a caller doing
/// readiness *and* the actual I/O can collapse both into one syscall's round trip per tick,
/// not one epoll_wait-equivalent plus one read()/write() syscall per ready fd.
pub struct IoUringDriver {
    ring_fd: OwnedFd,
    sq_ring: Mapping,
    cq_ring: Mapping,
    sqes: Mapping,
    sq_off: IoSqringOffsets,
    cq_off: IoCqringOffsets,
    sq_mask: u32,
    cq_mask: u32,
    sq_published_tail: u32, // last tail value the kernel has been told about
    sq_local_tail: u32,     // our own count of SQEs written (>= sq_published_tail)
    registered: Table<usize, Interest>,
    /// Buffers owned by an in-flight `submit_read`/`submit_write`, keyed by token — the
    /// io_uring "drop incompatibility" this module's own doc header names (a future dropped
    /// mid-flight while the kernel still writes into its buffer) dissolves exactly the way
    /// that header describes: the buffer's ownership moves in here BEFORE the kernel ever
    /// touches it, so nothing above this driver can drop or move it out from under an
    /// in-flight operation. `take_buffer` (backed by `Table::remove`) hands ownership back
    /// once the completion names it — the piece the header flagged as missing until now.
    bufs: Table<usize, Vec<u8>>,
    /// Whether the ring was set up with `IORING_SETUP_SQPOLL` — gates whether `flush()`/
    /// `wait()`'s submission step can skip `io_uring_enter` entirely (the kernel thread
    /// notices the tail update on its own) or must always call it (no thread, no other way
    /// to hand new SQEs to the kernel).
    sqpoll: bool,
    /// Provided-buffer pools registered via `register_recv_ring`, keyed by buffer-group id.
    buf_rings: Table<u16, BufRing>,
    /// Tokens with an armed multishot recv in flight, keyed by token → buffer-group id —
    /// what `wait()` checks (alongside `bufs`) to tell a recv completion apart from a plain
    /// poll's. Removed when a completion arrives without `IORING_CQE_F_MORE` (the multishot
    /// ended; `Readiness::ENDED` tells the caller to re-arm via `submit_recv_multishot`).
    recv_multishot: Table<usize, u16>,
    /// Queued completions per multishot-recv token — a queue, not a single slot, because
    /// (unlike one-shot `submit_read`) a multishot can generate several completions between
    /// two `wait()` calls; `recv_multishot_take` drains it FIFO. The third element is
    /// `IORING_CQE_F_SOCK_NONEMPTY` — see `recv_multishot_take`'s doc for why a caller must
    /// check it before treating `res == 0` as EOF.
    recv_queue: Table<usize, VecDeque<(i32, Vec<u8>, bool)>>,
    /// Count of real `io_uring_enter` syscalls this driver has issued — the direct telemetry
    /// for "how much batching is actually happening": divide completions reaped by this to
    /// get the real amortization factor, rather than inferring it from wall-clock alone.
    enter_calls: u64,
}

impl IoUringDriver {
    #[inline]
    unsafe fn sq_field(&self, byte_off: u32) -> &AtomicU32 {
        unsafe { &*(self.sq_ring.ptr.add(byte_off as usize) as *const AtomicU32) }
    }
    #[inline]
    unsafe fn cq_field(&self, byte_off: u32) -> &AtomicU32 {
        unsafe { &*(self.cq_ring.ptr.add(byte_off as usize) as *const AtomicU32) }
    }
    #[inline]
    unsafe fn sq_array_slot(&self, idx: u32) -> *mut u32 {
        let off = self.sq_off.array as usize + (idx as usize) * 4;
        unsafe { self.sq_ring.ptr.add(off) as *mut u32 }
    }
    #[inline]
    unsafe fn sqe_slot(&self, idx: u32) -> *mut IoUringSqe {
        unsafe { (self.sqes.ptr as *mut IoUringSqe).add(idx as usize) }
    }
    #[inline]
    unsafe fn cqe_slot(&self, idx: u32) -> *const IoUringCqe {
        let off = self.cq_off.cqes as usize + (idx as usize) * std::mem::size_of::<IoUringCqe>();
        unsafe { self.cq_ring.ptr.add(off) as *const IoUringCqe }
    }

    /// Write one SQE into the next free slot — not yet visible to the kernel until `wait()`
    /// publishes the new tail.
    fn push_sqe(&mut self, sqe: IoUringSqe) {
        let idx = self.sq_local_tail & self.sq_mask;
        // SAFETY: `idx` is within `[0, sq_entries)` by the mask; this slot is exclusively ours
        // until we publish a tail that covers it (the kernel only reads slots the published
        // tail names).
        unsafe {
            *self.sqe_slot(idx) = sqe;
            *self.sq_array_slot(idx) = idx;
        }
        self.sq_local_tail = self.sq_local_tail.wrapping_add(1);
    }

    fn poll_events(interest: Interest) -> u32 {
        let mut e = 0;
        if interest.is_readable() {
            e |= sys::POLLIN;
        }
        if interest.is_writable() {
            e |= sys::POLLOUT;
        }
        e
    }

    /// A negative `res` is this poll's own error (e.g. the fd closed underneath it) —
    /// surfaced as readable so the caller's next real read/write discovers and reports it,
    /// same philosophy as `EpollDriver`'s HUP/ERR-as-readable (see that driver's `from_flags`).
    fn from_poll_res(res: i32) -> Readiness {
        if res < 0 {
            return Readiness::READABLE;
        }
        let bits = res as u32;
        let mut r = Readiness::from_bits(0);
        if bits & (sys::POLLIN | sys::POLLHUP | sys::POLLERR) != 0 {
            r = r.add(Readiness::READABLE);
        }
        if bits & sys::POLLOUT != 0 {
            r = r.add(Readiness::WRITABLE);
        }
        r
    }

    fn submit_poll_add(&mut self, token: Token, fd: RawFd, interest: Interest) {
        self.push_sqe(IoUringSqe {
            opcode: sys::IORING_OP_POLL_ADD,
            fd,
            len: sys::IORING_POLL_ADD_MULTI, // POLL_ADD's command flags live in `len`, not a union tag
            poll32_events: Self::poll_events(interest),
            user_data: token.0 as u64,
            ..Default::default()
        });
    }

    fn submit_poll_remove(&mut self, token: Token) {
        self.push_sqe(IoUringSqe {
            opcode: sys::IORING_OP_POLL_REMOVE,
            fd: -1,
            addr: token.0 as u64, // the target: which existing poll's user_data to cancel
            user_data: REMOVE_SENTINEL,
            ..Default::default()
        });
    }

    /// Submit a read into `buf` — takes ownership of the buffer (stored in `self.bufs`,
    /// keyed by `token`) for exactly as long as the kernel might be writing into it, which is
    /// what makes the raw pointer below sound: the buffer can't be dropped, moved, or
    /// reallocated by anything above this driver until `take_buffer` hands it back after the
    /// completion names `token`. Result (byte count or a negative errno) reaches the caller
    /// via `Events::result` on the `wait()` call that reaps the completion. `token` must not
    /// already have a read/write in flight or a poll registered — this driver distinguishes
    /// completion kinds by "does this token have a buffer pending," so reusing one for both
    /// at once is a caller error, not something this driver can detect and reject cheaply.
    pub fn submit_read(&mut self, token: Token, fd: BorrowedFd<'_>, mut buf: Vec<u8>) {
        let ptr = buf.as_mut_ptr();
        let len = buf.len() as u32;
        self.bufs.put(token.0, buf); // moving a Vec doesn't move its heap allocation — `ptr` stays valid
        self.push_sqe(IoUringSqe {
            opcode: sys::IORING_OP_READ,
            fd: fd.as_raw_fd(),
            addr: ptr as u64,
            len,
            user_data: token.0 as u64,
            ..Default::default()
        });
    }

    /// Submit a write of `buf`'s contents — same ownership shape as `submit_read` (see its
    /// doc): the buffer moves into `self.bufs` before the kernel can read from it, and
    /// `take_buffer` returns it (bytes unchanged; a write doesn't consume the buffer) once
    /// the completion arrives.
    pub fn submit_write(&mut self, token: Token, fd: BorrowedFd<'_>, buf: Vec<u8>) {
        let ptr = buf.as_ptr();
        let len = buf.len() as u32;
        self.bufs.put(token.0, buf);
        self.push_sqe(IoUringSqe {
            opcode: sys::IORING_OP_WRITE,
            fd: fd.as_raw_fd(),
            addr: ptr as u64,
            len,
            user_data: token.0 as u64,
            ..Default::default()
        });
    }

    /// Reclaim a completed read/write's buffer — `None` if `token` has no pending or
    /// completed operation's buffer here (already taken, or never submitted).
    pub fn take_buffer(&mut self, token: Token) -> Option<Vec<u8>> {
        self.bufs.remove(&token.0)
    }

    /// Certify a pool of `entries` buffers (`buf_size` bytes each) for buffer-group `bgid` —
    /// ONCE. This is the B/U rest-state `submit_recv_multishot` reads against forever after:
    /// register a connection's (or the whole driver's) recv pool here, and every subsequent
    /// arrival on a multishot recv against `bgid` is a coordinate read (the CQE names which
    /// buffer id got the data, `wait()` copies it out and recycles the slot), never a fresh
    /// per-operation certification. `entries` must be a nonzero power of two (io_uring's own
    /// ring-mask requirement, same as the SQ/CQ rings this driver already sets up).
    pub fn register_recv_ring(&mut self, bgid: u16, entries: u16, buf_size: u32) -> io::Result<()> {
        if entries == 0 || !entries.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "register_recv_ring: entries must be a nonzero power of two",
            ));
        }
        // A caller keying `bgid` off a small, bounded value (e.g. `fd as u16`) will
        // legitimately see the SAME `bgid` come back once the kernel recycles that number
        // for a brand-new fd (verified live: a stale registration left EEXIST on the next
        // real connection's real registration attempt) — self-heal by tearing down any old
        // pool under this `bgid` first, rather than requiring every caller to remember to.
        if self.buf_rings.get(&bgid).is_some() {
            self.unregister_recv_ring(bgid)?;
        }
        let mask = entries - 1;
        let desc_len = entries as usize * std::mem::size_of::<sys::IoUringBufDesc>();
        // Page-aligned, anonymous — `IORING_REGISTER_PBUF_RING`'s own pin requires it (see
        // `BufRing`'s doc); a plain `Vec<IoUringBufDesc>` is not guaranteed page-aligned.
        let mem_ptr = sys::mmap_anon(desc_len)?;
        let mem = Mapping {
            ptr: mem_ptr,
            len: desc_len,
        };
        let mut data = vec![0u8; entries as usize * buf_size as usize];
        for bid in 0..entries {
            // SAFETY: `data` was just allocated at exactly `entries * buf_size` bytes above;
            // every slot's offset is within that region. `mem` is a fresh mmap of exactly
            // `desc_len` bytes, so slot `bid` (< entries) is in range.
            let addr = unsafe { data.as_mut_ptr().add(bid as usize * buf_size as usize) } as u64;
            unsafe {
                *(mem.ptr as *mut sys::IoUringBufDesc).add(bid as usize) = sys::IoUringBufDesc {
                    addr,
                    len: buf_size,
                    bid,
                    resv: 0,
                };
            }
        }
        // SAFETY: registration below is the first point the kernel can see this memory —
        // nothing races this write. Byte offset 14 within slot 0 is the kernel's overlaid
        // `tail` field (see `BufRing::recycle`); every buffer starts available, so `tail`
        // starts at `entries`.
        unsafe {
            let tail_ptr = mem.ptr.add(14) as *mut u16;
            *tail_ptr = entries;
        }

        let reg = sys::IoUringBufReg {
            ring_addr: mem.ptr as u64,
            ring_entries: entries as u32,
            bgid,
            flags: 0,
            min_left: 0,
            resv: [0; 5],
        };
        if let Err(e) = sys::io_uring_register(
            self.ring_fd.as_fd(),
            sys::IORING_REGISTER_PBUF_RING,
            &reg as *const sys::IoUringBufReg as *const u8,
            1,
        ) {
            sys::munmap(mem.ptr, mem.len);
            return Err(e);
        }

        self.buf_rings.put(
            bgid,
            BufRing {
                mem,
                data,
                mask,
                buf_size,
                local_tail: entries,
            },
        );
        Ok(())
    }

    /// Tear down `bgid`'s registered pool — the inverse of `register_recv_ring`. Needed
    /// before the SAME `bgid` can be registered again (see that method's doc); also called
    /// directly by a caller that knows a `bgid` will never be reused and wants the kernel
    /// registration released promptly rather than waiting for the whole ring to close.
    pub fn unregister_recv_ring(&mut self, bgid: u16) -> io::Result<()> {
        let reg = sys::IoUringBufReg {
            bgid,
            ..Default::default()
        };
        sys::io_uring_register(
            self.ring_fd.as_fd(),
            sys::IORING_UNREGISTER_PBUF_RING,
            &reg as *const sys::IoUringBufReg as *const u8,
            1,
        )?;
        self.buf_rings.remove(&bgid); // drops BufRing, which munmaps its descriptor ring
        Ok(())
    }

    /// Arm a multishot recv on `fd` under `token`, drawing buffers from `bgid`'s registered
    /// pool (`register_recv_ring` must have run first). ONE submission, MANY completions —
    /// each arrival is reaped by `wait()` and queued for `recv_multishot_take`, no
    /// resubmission between them. When a completion comes back without more following
    /// (`Readiness::ENDED` on this token), call this again to re-arm.
    pub fn submit_recv_multishot(
        &mut self,
        token: Token,
        fd: BorrowedFd<'_>,
        bgid: u16,
    ) -> io::Result<()> {
        if self.buf_rings.get(&bgid).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "submit_recv_multishot: bgid not registered",
            ));
        }
        self.recv_multishot.put(token.0, bgid);
        self.push_sqe(IoUringSqe {
            opcode: sys::IORING_OP_RECV,
            flags: sys::IOSQE_BUFFER_SELECT,
            ioprio: sys::IORING_RECV_MULTISHOT,
            fd: fd.as_raw_fd(),
            // `len` MUST be 0 for plain multishot RECV (verified live: a nonzero len here
            // is EINVAL on this kernel) — the kernel caps each completion by whichever
            // buffer's own registered length got picked, not a single op-wide length.
            len: 0,
            buf_index_or_group: bgid,
            user_data: token.0 as u64,
            ..Default::default()
        });
        Ok(())
    }

    /// Pop one queued multishot-recv completion for `token`, FIFO order: `(res, data,
    /// sock_nonempty)` — byte count/errno, the bytes already copied out of the kernel-shared
    /// buffer ring, and `IORING_CQE_F_SOCK_NONEMPTY`. A queue, not a single slot, because a
    /// multishot can generate several completions between two `wait()` calls — unlike
    /// `take_buffer`, which is safe as a single slot only because one-shot `submit_read` can
    /// never have more than one pending completion per token.
    ///
    /// **`res == 0` is NOT always EOF** — verified live, the exact bug this field exists to
    /// prevent: when the pool runs out of buffers for one attempt, the kernel can end the
    /// multishot with `res == 0` while `sock_nonempty` is set, meaning the socket still has
    /// data waiting; only `res == 0 && !sock_nonempty` is a genuine peer close. A caller that
    /// treats every `res == 0` as EOF silently truncates real transfers under concurrency.
    pub fn recv_multishot_take(&mut self, token: Token) -> Option<(i32, Vec<u8>, bool)> {
        let mut q = self.recv_queue.remove(&token.0)?;
        let item = q.pop_front();
        if !q.is_empty() {
            self.recv_queue.put(token.0, q);
        }
        item
    }

    /// Whether `token`'s multishot recv is still armed. Mirrors `Readiness::ENDED`: once a
    /// completion arrives with no more following, this goes false until re-armed.
    pub fn recv_multishot_active(&self, token: Token) -> bool {
        self.recv_multishot.get(&token.0).is_some()
    }
}

impl Drop for IoUringDriver {
    fn drop(&mut self) {
        sys::munmap(self.sqes.ptr, self.sqes.len);
        sys::munmap(self.sq_ring.ptr, self.sq_ring.len); // single mmap: sq_ring == cq_ring region
    }
}

impl IoUringDriver {
    /// Set up the ring (`io_uring_setup`) and mmap the SQ/CQ rings + SQE array. `capacity` is
    /// the requested submission-queue depth (rounded up to a power of two by the kernel; the
    /// actual size comes back in `params` and is what this driver uses from then on).
    ///
    /// Requires `IORING_FEAT_SINGLE_MMAP` (the SQ and CQ ring headers share one mmap region) —
    /// universal since Linux 5.4 (2019); a kernel old enough to lack it gets a plain
    /// `Unsupported` error here rather than a second, untested mmap-layout code path (labeled
    /// scope narrowing, Axiom XIII, not a silent gap).
    pub fn new(capacity: usize) -> io::Result<Self> {
        Self::new_with_params(capacity, 0, 0)
    }

    /// Same as [`Self::new`], but with a kernel-side thread polling the SQ ring
    /// (`IORING_SETUP_SQPOLL`) — submitting new work (`flush()`, or `wait()`'s own submit
    /// step) then costs no syscall at all as long as the thread is still awake; it only goes
    /// idle after `idle_ms` milliseconds with nothing new to process, at which point exactly
    /// one wakeup call is needed to bring it back (`flush()`/`wait()` detect and pay this
    /// automatically via `IORING_SQ_NEED_WAKEUP`). The tradeoff this makes explicit rather
    /// than hiding: that thread genuinely runs, consuming real CPU while polling — pick
    /// `idle_ms` for the actual gap between bursts of work, not "as long as possible."
    ///
    /// No elevated capability needed for this unpinned form (`IORING_SETUP_SQ_AFF`, CPU-
    /// pinning the thread, is the part that wants `CAP_SYS_NICE`, and this driver doesn't set
    /// it) — verified live against this machine's kernel, not assumed from documentation.
    pub fn new_sqpoll(capacity: usize, idle_ms: u32) -> io::Result<Self> {
        Self::new_with_params(capacity, sys::IORING_SETUP_SQPOLL, idle_ms)
    }

    fn new_with_params(capacity: usize, setup_flags: u32, sq_thread_idle: u32) -> io::Result<Self> {
        let mut params = IoUringParams {
            flags: setup_flags,
            sq_thread_idle,
            ..Default::default()
        };
        let ring_fd = sys::io_uring_setup(capacity.max(1) as u32, &mut params)?;

        if params.features & sys::IORING_FEAT_SINGLE_MMAP == 0 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "IoUringDriver requires IORING_FEAT_SINGLE_MMAP (Linux 5.4+)",
            ));
        }

        let sq_off = params.sq_off;
        let cq_off = params.cq_off;
        let cqe_size = std::mem::size_of::<IoUringCqe>() as u32;
        let ring_len = (sq_off.array + params.sq_entries * 4)
            .max(cq_off.cqes + params.cq_entries * cqe_size) as usize;

        let prot = sys::PROT_READ | sys::PROT_WRITE;
        let ring_ptr = sys::mmap(ring_len, prot, ring_fd.as_fd(), sys::IORING_OFF_SQ_RING)?;
        let sqes_len = params.sq_entries as usize * std::mem::size_of::<IoUringSqe>();
        let sqes_ptr = match sys::mmap(sqes_len, prot, ring_fd.as_fd(), sys::IORING_OFF_SQES) {
            Ok(p) => p,
            Err(e) => {
                sys::munmap(ring_ptr, ring_len);
                return Err(e);
            }
        };

        // SAFETY: both reads are within the just-mapped `ring_len`-byte region (the offsets
        // themselves come from the kernel's own `params.sq_off`/`cq_off`, always in range).
        let sq_mask = unsafe { *(ring_ptr.add(sq_off.ring_mask as usize) as *const u32) };
        let cq_mask = unsafe { *(ring_ptr.add(cq_off.ring_mask as usize) as *const u32) };

        Ok(IoUringDriver {
            ring_fd,
            sq_ring: Mapping {
                ptr: ring_ptr,
                len: ring_len,
            },
            cq_ring: Mapping {
                ptr: ring_ptr,
                len: ring_len,
            }, // single mmap: same region, both faces
            sqes: Mapping {
                ptr: sqes_ptr,
                len: sqes_len,
            },
            sq_off,
            cq_off,
            sq_mask,
            cq_mask,
            sq_published_tail: 0,
            sq_local_tail: 0,
            registered: Table::new(),
            bufs: Table::new(),
            sqpoll: setup_flags & sys::IORING_SETUP_SQPOLL != 0,
            buf_rings: Table::new(),
            recv_multishot: Table::new(),
            recv_queue: Table::new(),
            enter_calls: 0,
        })
    }

    /// Publish every SQE written since the last flush/wait — without waiting for or reaping
    /// any completions. On a plain ring this still costs one `io_uring_enter` (there's no
    /// other way to hand new SQEs to the kernel); on an `SQPOLL` ring, this is a plain memory
    /// write and costs NO syscall at all as long as the polling thread is still awake — the
    /// actual lever `new_sqpoll` exists for: submit now, do other work, `wait()` later, and
    /// the kernel processes the submission concurrently with that other work instead of only
    /// starting once a blocking wait() call gets around to it.
    pub fn flush(&mut self) -> io::Result<()> {
        let to_submit = self.sq_local_tail.wrapping_sub(self.sq_published_tail);
        if to_submit == 0 {
            return Ok(());
        }
        // SAFETY: Release store publishes every prior write into the SQE/array slots below
        // the new tail — the kernel (or its polling thread) Acquire-loads this same field
        // before reading them.
        unsafe {
            self.sq_field(self.sq_off.tail)
                .store(self.sq_local_tail, Ordering::Release)
        };
        self.sq_published_tail = self.sq_local_tail;

        if !self.sqpoll {
            sys::io_uring_enter(self.ring_fd.as_fd(), to_submit, 0, 0)?;
            self.enter_calls += 1;
            return Ok(());
        }
        // SAFETY: Acquire load of the kernel-owned flags field — pairs with the kernel's
        // Release store of NEED_WAKEUP when its polling thread goes to sleep.
        let flags = unsafe { self.sq_field(self.sq_off.flags).load(Ordering::Acquire) };
        if flags & sys::IORING_SQ_NEED_WAKEUP != 0 {
            sys::io_uring_enter(self.ring_fd.as_fd(), 0, 0, sys::IORING_ENTER_SQ_WAKEUP)?;
            self.enter_calls += 1;
        }
        Ok(())
    }

    /// Count of real `io_uring_enter` syscalls issued so far — divide completions reaped by
    /// this for the actual amortization factor a caller is achieving, not an assumed one.
    pub fn enter_calls(&self) -> u64 {
        self.enter_calls
    }
}

impl Driver for IoUringDriver {
    fn register(&mut self, token: Token, fd: BorrowedFd<'_>, interest: Interest) -> io::Result<()> {
        self.registered
            .insert_with(token.0, interest, |new, _old| new);
        self.submit_poll_add(token, fd.as_raw_fd(), interest);
        Ok(())
    }

    /// mio semantics: reregister REPLACES interest. io_uring has no atomic "modify this poll's
    /// mask" this driver relies on — cancel the old multishot poll and arm a fresh one, the
    /// same two-step shape a caller doing this over raw `epoll_ctl(MOD)` gets internally
    /// anyway; both SQEs ride in the same next `wait()`'s submission batch.
    fn reregister(
        &mut self,
        token: Token,
        fd: BorrowedFd<'_>,
        interest: Interest,
    ) -> io::Result<()> {
        self.registered
            .insert_with(token.0, interest, |new, _old| new);
        self.submit_poll_remove(token);
        self.submit_poll_add(token, fd.as_raw_fd(), interest);
        Ok(())
    }

    fn deregister(&mut self, token: Token, _fd: BorrowedFd<'_>) -> io::Result<()> {
        self.registered
            .insert_with(token.0, Interest::default(), |new, _old| new);
        self.submit_poll_remove(token);
        Ok(())
    }

    fn wait(&mut self, events: &mut Events, timeout_ms: Option<u32>) -> io::Result<usize> {
        // No native "block up to N ms" on the raw io_uring_enter path without a kernel timeout
        // SQE (IORING_OP_TIMEOUT) linked ahead of the wait — out of scope for this pass
        // (labeled, Axiom XIII): `None` blocks for at least one completion; `Some(0)` polls
        // without blocking; any other `Some(ms)` degrades to a non-blocking check today rather
        // than a real bounded wait.
        let min_complete = match timeout_ms {
            None => 1,
            Some(_) => 0,
        };
        let to_submit = self.sq_local_tail.wrapping_sub(self.sq_published_tail);

        if to_submit > 0 || min_complete > 0 {
            // Either there's new work to hand the kernel, or the caller actually wants to
            // block — both need a real syscall crossing. When neither is true (a non-
            // blocking check with nothing new queued), completions the kernel already wrote
            // are sitting in the CQ ring's shared memory regardless of whether *this* call
            // asks for them — read them directly below, zero syscalls, the same way a
            // caller checking twice in a row without submitting anything new always could.
            if self.sqpoll {
                // SQPOLL: publish-and-maybe-wake is `flush()`'s job (a plain memory write
                // when the thread's awake, no separate enter() for submission at all); only
                // the actual block-for-completions step, if requested, needs its own call.
                self.flush()?;
                if min_complete > 0 {
                    sys::io_uring_enter(
                        self.ring_fd.as_fd(),
                        0,
                        min_complete,
                        sys::IORING_ENTER_GETEVENTS,
                    )?;
                    self.enter_calls += 1;
                }
            } else {
                // No polling thread — submission and (if requested) blocking-for-completions
                // collapse into the one call this driver has always made here.
                if to_submit > 0 {
                    // SAFETY: Release store publishes every prior write into the SQE/array
                    // slots below the new tail — the kernel Acquire-loads this same field
                    // before reading them.
                    unsafe {
                        self.sq_field(self.sq_off.tail)
                            .store(self.sq_local_tail, Ordering::Release)
                    };
                    self.sq_published_tail = self.sq_local_tail;
                }
                sys::io_uring_enter(
                    self.ring_fd.as_fd(),
                    to_submit,
                    min_complete,
                    sys::IORING_ENTER_GETEVENTS,
                )?;
                self.enter_calls += 1;
            }
        }

        let head = unsafe { self.cq_field(self.cq_off.head).load(Ordering::Relaxed) };
        // SAFETY: Acquire load — pairs with the kernel's Release store of the tail after it
        // finishes writing the CQE(s) below it, so every CQE this reads below is visible.
        let tail = unsafe { self.cq_field(self.cq_off.tail).load(Ordering::Acquire) };

        let mut n = 0;
        let mut i = head;
        while i != tail {
            let idx = i & self.cq_mask;
            // SAFETY: `idx` is within the CQ ring's bounds by the mask; the Acquire load above
            // makes this slot's contents visible.
            let cqe = unsafe { *self.cqe_slot(idx) };
            if cqe.user_data != REMOVE_SENTINEL {
                let token = Token(cqe.user_data as usize);
                if let Some(&bgid) = self.recv_multishot.get(&token.0) {
                    // A multishot recv completion: copy the data out of the shared buffer
                    // ring immediately (streaming law — read this arrival at its coordinate,
                    // the buffer id the CQE names) and recycle the slot right away, so the
                    // SAME certified pool serves the next arrival with no caller round trip.
                    let more = cqe.flags & sys::IORING_CQE_F_MORE != 0;
                    let sock_nonempty = cqe.flags & sys::IORING_CQE_F_SOCK_NONEMPTY != 0;
                    let item = if cqe.res >= 0 && cqe.flags & sys::IORING_CQE_F_BUFFER != 0 {
                        let buf_id = (cqe.flags >> 16) as u16;
                        match self.buf_rings.remove(&bgid) {
                            Some(mut ring) => {
                                let chunk = ring.buf_data(buf_id, cqe.res as usize).to_vec();
                                ring.recycle(buf_id);
                                self.buf_rings.put(bgid, ring);
                                (cqe.res, chunk, sock_nonempty)
                            }
                            None => (cqe.res, Vec::new(), sock_nonempty),
                        }
                    } else {
                        (cqe.res, Vec::new(), sock_nonempty)
                    };
                    self.recv_queue
                        .insert_with(token.0, VecDeque::from([item]), |new, mut old| {
                            old.extend(new);
                            old
                        });
                    if !more {
                        self.recv_multishot.remove(&token.0);
                    }
                    events.record(
                        token,
                        if more {
                            Readiness::DONE
                        } else {
                            Readiness::DONE.add(Readiness::ENDED)
                        },
                    );
                } else if self.bufs.get(&token.0).is_some() {
                    // A pending buffer means this token names a read/write, not a poll — the
                    // result is a byte count or a negative errno, not a readiness mask.
                    events.record_result(token, cqe.res);
                    events.record(token, Readiness::DONE);
                } else {
                    events.record(token, Self::from_poll_res(cqe.res));
                }
                n += 1;
            }
            i = i.wrapping_add(1);
        }
        // SAFETY: Release store — tells the kernel these CQE slots are free to reuse; must
        // happen after every read of them above.
        unsafe {
            self.cq_field(self.cq_off.head)
                .store(tail, Ordering::Release)
        };

        Ok(n)
    }
}

// ── The real boundary: io_uring over a real pipe, our syscalls, no liburing. ──
#[cfg(test)]
mod tests {
    //! Was `mod tests { .. }` inline inside `iouring_driver`; still its child, so `super` is the
    //! driver and `super::super` is the reactor model — both `use` lines below are unchanged.

    use super::super::{Driver, Events, Interest, Reactor, Readiness, Token};
    use super::{sys, IoUringDriver};
    use std::os::fd::AsFd;

    #[test]
    fn io_uring_reports_a_readable_pipe() {
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut reactor = Reactor::new(IoUringDriver::new(16).expect("io_uring_setup"));
        reactor
            .register(Token(1), rd.as_fd(), Interest::READABLE)
            .unwrap();

        let mut events = Events::new();
        assert_eq!(
            reactor.poll(&mut events, Some(0)).unwrap(),
            0,
            "empty pipe not readable"
        );

        sys::write(wr.as_fd(), b"x").expect("write");
        let n = reactor.poll(&mut events, None).unwrap();
        assert_eq!(n, 1, "io_uring reported the readable pipe");
        assert!(
            events.get(Token(1)).is_readable(),
            "token 1 demuxed to readable"
        );
        let _ = Readiness::DONE; // (completion face shares this demux; exercised above)

        reactor.deregister(Token(1), rd.as_fd()).unwrap();
    }

    #[test]
    fn io_uring_multishot_reports_repeatedly_without_resubmission() {
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut reactor = Reactor::new(IoUringDriver::new(16).expect("io_uring_setup"));
        reactor
            .register(Token(5), rd.as_fd(), Interest::READABLE)
            .unwrap();

        let mut events = Events::new();
        sys::write(wr.as_fd(), b"a").expect("write");
        assert_eq!(reactor.poll(&mut events, None).unwrap(), 1);
        assert!(events.get(Token(5)).is_readable());

        // Drain the pipe so it goes back to not-readable, then write again — the SAME
        // registration (no re-register call) must report readiness a second time, proving the
        // multishot poll is still armed rather than a one-shot that silently stopped.
        let mut buf = [0u8; 1];
        sys::read(rd.as_fd(), &mut buf).expect("read");
        sys::write(wr.as_fd(), b"b").expect("write");
        let n = reactor.poll(&mut events, None).unwrap();
        assert_eq!(
            n, 1,
            "multishot poll reported a second time with no resubmission"
        );
        assert!(events.get(Token(5)).is_readable());
    }

    #[test]
    fn io_uring_deregister_stops_further_reports() {
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut reactor = Reactor::new(IoUringDriver::new(16).expect("io_uring_setup"));
        let mut events = Events::new();
        reactor
            .register(Token(9), rd.as_fd(), Interest::READABLE)
            .unwrap();
        // Actually submit and arm the poll on a not-yet-readable pipe first — deregistering
        // in the SAME unsubmitted batch as the register races the kernel's own immediate
        // readiness check against the cancel, which is a genuine ordering ambiguity in the
        // test (not the driver): a POLL_ADD checks current readiness as soon as it's
        // processed, and if it's already satisfied by the time a same-batch POLL_REMOVE runs,
        // there's nothing left to cancel. Two `wait()` calls, like any real caller's reactor
        // loop, avoids that race entirely.
        reactor.poll(&mut events, Some(0)).unwrap();
        reactor.deregister(Token(9), rd.as_fd()).unwrap();
        reactor.poll(&mut events, Some(0)).unwrap();

        sys::write(wr.as_fd(), b"x").expect("write");
        let n = reactor.poll(&mut events, Some(0)).unwrap();
        assert_eq!(n, 0, "deregistered token must not report");
    }

    #[test]
    fn io_uring_reregister_replaces_interest() {
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut reactor = Reactor::new(IoUringDriver::new(16).expect("io_uring_setup"));
        let mut events = Events::new();
        // WRITABLE on a pipe's READ end never becomes ready — arms and stays pending, so the
        // reregister below has something genuinely live to replace (see the same-batch-race
        // note on the deregister test above for why this needs its own `wait()` first).
        reactor
            .register(Token(2), rd.as_fd(), Interest::WRITABLE)
            .unwrap();
        reactor.poll(&mut events, Some(0)).unwrap();
        reactor
            .reregister(Token(2), rd.as_fd(), Interest::READABLE)
            .unwrap();
        reactor.poll(&mut events, Some(0)).unwrap();

        sys::write(wr.as_fd(), b"x").expect("write");
        let n = reactor.poll(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert!(
            events.get(Token(2)).is_readable(),
            "reregistered interest took effect"
        );
    }

    #[test]
    fn io_uring_sq_ring_wraparound_survives_many_cycles() {
        // A ring of 4 entries, driven through 50 register/wait/deregister/wait cycles — the
        // local SQ tail wraps past `sq_entries` many times over. If the index masking
        // (`idx & sq_mask`) or the published-vs-local tail bookkeeping were wrong, this would
        // either panic (out-of-bounds via the mask) or silently corrupt a still-in-flight SQE
        // slot the kernel hasn't read yet.
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut reactor = Reactor::new(IoUringDriver::new(4).expect("io_uring_setup"));
        let mut events = Events::new();
        for i in 0..50u64 {
            let token = Token(i as usize);
            reactor
                .register(token, rd.as_fd(), Interest::READABLE)
                .unwrap();
            reactor.poll(&mut events, Some(0)).unwrap();
            reactor.deregister(token, rd.as_fd()).unwrap();
            reactor.poll(&mut events, Some(0)).unwrap();
        }
        // The driver is still usable after 100 submissions through a 4-entry ring.
        reactor
            .register(Token(999), rd.as_fd(), Interest::READABLE)
            .unwrap();
        sys::write(wr.as_fd(), b"x").expect("write");
        let n = reactor.poll(&mut events, None).unwrap();
        assert_eq!(n, 1, "driver still correct after many wraparounds");
        assert!(events.get(Token(999)).is_readable());
    }

    #[test]
    fn io_uring_driver_leaks_no_fds() {
        // Compare the SLOPE across two rounds, not a single before/after snapshot — other
        // tests running concurrently in this same process open and close their own fds the
        // whole time (real noise, not a leak), but a genuine per-iteration leak would grow
        // roughly LINEARLY with the loop count while that ambient noise stays roughly flat.
        let count = || std::fs::read_dir("/proc/self/fd").unwrap().count();
        for _ in 0..50 {
            drop(IoUringDriver::new(16).expect("io_uring_setup"));
        }
        let after_round_1 = count();
        for _ in 0..50 {
            drop(IoUringDriver::new(16).expect("io_uring_setup"));
        }
        let after_round_2 = count();
        let delta = after_round_2 as i64 - after_round_1 as i64;
        assert!(
            delta.abs() <= 10,
            "fd count moved by {delta} across a second identical 50-cycle round \
             ({after_round_1} -> {after_round_2}) — a real per-iteration leak would scale with the loop count"
        );
    }

    #[test]
    fn io_uring_sqpoll_basic_readiness_and_readwrite() {
        // Same correctness contract as the plain ring, verified live rather than assumed:
        // SQPOLL setup, poll readiness, and read/write submission all still work correctly
        // with a kernel-side polling thread in the picture.
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut reactor = Reactor::new(IoUringDriver::new_sqpoll(16, 200).expect("SQPOLL setup"));
        reactor
            .register(Token(1), rd.as_fd(), Interest::READABLE)
            .unwrap();
        let mut events = Events::new();
        assert_eq!(reactor.poll(&mut events, Some(0)).unwrap(), 0);

        sys::write(wr.as_fd(), b"x").expect("write");
        let n = reactor.poll(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert!(events.get(Token(1)).is_readable());

        reactor
            .driver()
            .submit_read(Token(2), rd.as_fd(), vec![0u8; 16]);
        let n = reactor.poll(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            events.result(Token(2)),
            Some(1),
            "read must report the 1 byte written above"
        );
        let buf = reactor.driver().take_buffer(Token(2)).unwrap();
        assert_eq!(&buf[..1], b"x");
    }

    #[test]
    fn io_uring_sqpoll_flush_processes_in_the_background() {
        // The actual claim `new_sqpoll` exists for: submit + flush() (no blocking wait), then
        // real wall-clock time passes while this thread does something else, and the kernel's
        // polling thread has already processed the read by the time a LATER, purely-reaping
        // wait(Some(0)) call checks — proving the work happened concurrently, not only once a
        // blocking wait() call got around to driving it.
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut driver = IoUringDriver::new_sqpoll(16, 500).expect("SQPOLL setup");
        sys::write(wr.as_fd(), b"hello").expect("write");

        driver.submit_read(Token(1), rd.as_fd(), vec![0u8; 16]);
        driver.flush().expect("flush (submission only, no wait)");

        // Give the kernel's polling thread real wall-clock time to notice and process the
        // read on its own, with this thread doing nothing that would drive it forward.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let mut events = Events::new();
        let n = driver
            .wait(&mut events, Some(0))
            .expect("reap only, nothing new to submit");
        assert_eq!(
            n, 1,
            "the read completed via the background polling thread, not this wait() call"
        );
        assert_eq!(events.result(Token(1)), Some(5));
        let buf = driver.take_buffer(Token(1)).unwrap();
        assert_eq!(&buf[..5], b"hello");
    }

    #[test]
    fn io_uring_sqpoll_wakes_after_thread_goes_idle() {
        // A short idle timeout, then sleep past it — the polling thread genuinely goes to
        // sleep (IORING_SQ_NEED_WAKEUP gets set), and a submission afterward must still
        // complete correctly via the explicit wakeup path in `flush()`/`wait()`, not silently
        // stall forever waiting for a thread that's no longer polling.
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut driver = IoUringDriver::new_sqpoll(16, 20).expect("SQPOLL setup");
        std::thread::sleep(std::time::Duration::from_millis(150)); // thread goes idle

        sys::write(wr.as_fd(), b"woken").expect("write");
        driver.submit_read(Token(1), rd.as_fd(), vec![0u8; 16]);
        let mut events = Events::new();
        let n = driver
            .wait(&mut events, None)
            .expect("wakeup + reap after the thread went idle");
        assert_eq!(n, 1);
        assert_eq!(events.result(Token(1)), Some(5));
        let buf = driver.take_buffer(Token(1)).unwrap();
        assert_eq!(&buf[..5], b"woken");
    }

    #[test]
    fn io_uring_submit_read_gets_the_real_bytes_and_count() {
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut driver = IoUringDriver::new(16).expect("io_uring_setup");
        sys::write(wr.as_fd(), b"hello").expect("write");

        driver.submit_read(Token(1), rd.as_fd(), vec![0u8; 16]);
        let mut events = Events::new();
        let n = driver.wait(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            events.result(Token(1)),
            Some(5),
            "read must report exactly 5 bytes"
        );
        let buf = driver
            .take_buffer(Token(1))
            .expect("buffer returned after completion");
        assert_eq!(
            &buf[..5],
            b"hello",
            "the buffer must contain the actual bytes read"
        );
        assert!(
            driver.take_buffer(Token(1)).is_none(),
            "buffer is gone after being taken once"
        );
    }

    #[test]
    fn io_uring_submit_write_lands_the_real_bytes_on_the_wire() {
        let (rd, wr) = sys::pipe2_cloexec().expect("pipe2");
        let mut driver = IoUringDriver::new(16).expect("io_uring_setup");

        driver.submit_write(Token(2), wr.as_fd(), b"world".to_vec());
        let mut events = Events::new();
        let n = driver.wait(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            events.result(Token(2)),
            Some(5),
            "write must report exactly 5 bytes"
        );
        let buf = driver
            .take_buffer(Token(2))
            .expect("buffer returned after completion");
        assert_eq!(buf, b"world", "write doesn't consume the buffer's contents");

        // The real proof: read the bytes back with a plain syscall, independent of io_uring.
        let mut got = [0u8; 5];
        sys::read(rd.as_fd(), &mut got).expect("read");
        assert_eq!(&got, b"world", "the write actually reached the pipe");
    }

    #[test]
    fn io_uring_many_reads_and_writes_batch_through_one_wait() {
        // N independent pipes, N reads and N writes all submitted before a SINGLE wait() call
        // — the actual claim this rung exists to prove: many I/O operations collapse into one
        // io_uring_enter, not one syscall per operation.
        const N: usize = 32;
        let pairs: Vec<_> = (0..N)
            .map(|_| sys::pipe2_cloexec().expect("pipe2"))
            .collect();
        let mut driver = IoUringDriver::new(64).expect("io_uring_setup");

        for (i, (_, wr)) in pairs.iter().enumerate() {
            sys::write(wr.as_fd(), format!("msg{i}").as_bytes()).expect("write");
        }
        for (i, (rd, _)) in pairs.iter().enumerate() {
            driver.submit_read(Token(i), rd.as_fd(), vec![0u8; 16]);
        }
        let mut events = Events::new();
        let n = driver.wait(&mut events, None).unwrap();
        assert_eq!(n, N, "all N reads complete from one wait() call");
        for i in 0..N {
            let expect = format!("msg{i}");
            let res = events.result(Token(i)).unwrap();
            assert_eq!(res as usize, expect.len());
            let buf = driver.take_buffer(Token(i)).unwrap();
            assert_eq!(&buf[..res as usize], expect.as_bytes());
        }
    }

    #[test]
    fn multishot_recv_reports_without_resubmission() {
        // The actual claim this rung exists to prove: ONE submit_recv_multishot, TWO
        // separate arrivals reaped across two wait() calls, no resubmission in between —
        // unlike submit_read, which needs a fresh submission per operation.
        let (a, b) = sys::socketpair().expect("socketpair");
        let mut driver = IoUringDriver::new(16).expect("io_uring_setup");
        driver
            .register_recv_ring(1, 4, 64)
            .expect("register_recv_ring");
        driver
            .submit_recv_multishot(Token(1), a.as_fd(), 1)
            .expect("submit_recv_multishot");

        sys::write(b.as_fd(), b"first").expect("write");
        let mut events = Events::new();
        let n = driver.wait(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert!(events.get(Token(1)).is_done());
        assert!(
            !events.get(Token(1)).is_ended(),
            "a live stream read must not end the multishot"
        );
        assert!(
            driver.recv_multishot_active(Token(1)),
            "still armed with no resubmission"
        );
        let (res, buf, _) = driver
            .recv_multishot_take(Token(1))
            .expect("queued completion");
        assert_eq!(res, 5);
        assert_eq!(&buf, b"first");
        assert!(driver.recv_multishot_take(Token(1)).is_none(), "drained");

        // Second arrival, same submission — no submit_recv_multishot call in between.
        sys::write(b.as_fd(), b"second!").expect("write");
        let n = driver.wait(&mut events, None).unwrap();
        assert_eq!(n, 1);
        let (res, buf, _) = driver
            .recv_multishot_take(Token(1))
            .expect("second queued completion");
        assert_eq!(res, 7);
        assert_eq!(&buf, b"second!");
    }

    #[test]
    fn multishot_recv_queues_multiple_completions_between_waits() {
        // Several arrivals land before a single wait() call drains them — recv_multishot_take
        // must return every one of them, FIFO, not just the last (an overwriting single slot
        // would silently lose data here).
        let (a, b) = sys::socketpair().expect("socketpair");
        let mut driver = IoUringDriver::new(16).expect("io_uring_setup");
        driver
            .register_recv_ring(2, 8, 32)
            .expect("register_recv_ring");
        driver
            .submit_recv_multishot(Token(7), a.as_fd(), 2)
            .expect("submit_recv_multishot");

        // Give the kernel a chance to see each write as a separate recv completion by
        // flushing between them (best-effort — TCP/unix stream sockets may still coalesce;
        // the assertion below checks total bytes recovered, not completion count).
        for msg in [&b"one"[..], &b"two"[..], &b"three"[..]] {
            sys::write(b.as_fd(), msg).expect("write");
        }
        let mut events = Events::new();
        driver.wait(&mut events, None).unwrap();

        let mut got = Vec::new();
        while let Some((res, buf, _)) = driver.recv_multishot_take(Token(7)) {
            assert!(res >= 0);
            got.extend_from_slice(&buf);
        }
        assert_eq!(
            got, b"onetwothree",
            "every queued arrival recovered, none dropped"
        );
    }

    #[test]
    fn multishot_recv_ends_on_eof_and_reports_readiness_ended() {
        let (a, b) = sys::socketpair().expect("socketpair");
        let mut driver = IoUringDriver::new(16).expect("io_uring_setup");
        driver
            .register_recv_ring(3, 4, 64)
            .expect("register_recv_ring");
        driver
            .submit_recv_multishot(Token(9), a.as_fd(), 3)
            .expect("submit_recv_multishot");

        drop(b); // peer hangs up — the next recv completion is EOF (res == 0)
        let mut events = Events::new();
        let n = driver.wait(&mut events, None).unwrap();
        assert_eq!(n, 1);
        assert!(
            events.get(Token(9)).is_ended(),
            "EOF must end the multishot"
        );
        assert!(
            !driver.recv_multishot_active(Token(9)),
            "must need re-arming after EOF"
        );
        let (res, _, _) = driver
            .recv_multishot_take(Token(9))
            .expect("EOF completion queued");
        assert_eq!(res, 0, "EOF reports 0 bytes");
    }

    #[test]
    fn register_recv_ring_self_heals_a_reused_bgid() {
        // A real scenario, not hypothetical: a caller keying `bgid` off a small bounded value
        // (e.g. `fd as u16` in tokio-xor) will see the kernel legitimately recycle that number
        // for a brand-new connection once the old one closes. Re-registering the SAME `bgid`
        // without tearing down the old pool first is `EEXIST` — this is the exact failure a
        // real integration hit, root-caused by tracing actual fd/inode numbers rather than
        // guessed at as a scheduler bug.
        let (a1, b1) = sys::socketpair().expect("socketpair 1");
        let mut driver = IoUringDriver::new(16).expect("io_uring_setup");
        driver
            .register_recv_ring(1, 4, 64)
            .expect("first register_recv_ring");
        driver
            .submit_recv_multishot(Token(1), a1.as_fd(), 1)
            .expect("submit 1");
        sys::write(b1.as_fd(), b"first-conn").expect("write 1");
        let mut events = Events::new();
        driver.wait(&mut events, None).unwrap();
        let (res, buf, _) = driver
            .recv_multishot_take(Token(1))
            .expect("first conn's data");
        assert_eq!(res as usize, 10);
        assert_eq!(&buf, b"first-conn");
        drop(a1);
        drop(b1); // both ends gone — as far as this driver's concerned, done with bgid 1

        // A second, unrelated connection reuses bgid 1 WITHOUT calling unregister_recv_ring
        // first — this must succeed (self-heal), not return the EEXIST the bug produced.
        let (a2, b2) = sys::socketpair().expect("socketpair 2");
        driver
            .register_recv_ring(1, 4, 64)
            .expect("register_recv_ring must self-heal a reused bgid, not EEXIST");
        driver
            .submit_recv_multishot(Token(2), a2.as_fd(), 1)
            .expect("submit 2");
        sys::write(b2.as_fd(), b"second-conn").expect("write 2");
        let mut events = Events::new();
        driver.wait(&mut events, None).unwrap();
        let (res, buf, _) = driver
            .recv_multishot_take(Token(2))
            .expect("second conn's data");
        assert_eq!(res as usize, 11);
        assert_eq!(
            &buf, b"second-conn",
            "the NEW pool must serve the NEW connection's own data"
        );
    }
}
