//! runtime — a current-thread async executor on the reactor (the **tokio-xor** layer).
//!
//! ## Where this sits
//!
//! ```text
//! hashtrinity Table  (+ io: reactor, proc: universal_lift/fromTerm, serde: codec)
//!   └─ reactor        mio / io_uring subsumed; the epoll cut (our own syscall)
//!        └─ runtime   tokio subsumed  ← THIS
//!             └─ (sanguine http)      hyper / axum, next
//! ```
//!
//! ## The whole thing is still one Table
//!
//! A task is a continuation — an open boundary closed by `apply`
//! (`boundary_conditions_not_lambda`). The **wake registry is one more token-keyed [`Table`]**:
//! `Table<usize, Waker>`. The reactor demuxes a ready token; we read *that* token's waker at its
//! coordinate and fire it. There is no scan of a task list — each ready task is read at its
//! coordinate (the streaming law), exactly as `reactor` reads each ready fd. readiness (epoll) and
//! completion (io_uring) are the same demux, so the same executor drives both backends.
//!
//! This first rung is single-task `block_on` (spawn / multi-task is the next rung) — enough to drive
//! a leaf async I/O future through the reactor. Zero `unsafe`: all of it is in the syscall cut below
//! `reactor`.

use crate::reactor::{EpollDriver, Events, Interest, Reactor, Token};
use crate::Table;
use std::cell::RefCell;
use std::future::Future;
use std::io;
use std::os::fd::AsFd;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

thread_local! {
    /// The running runtime for this thread — installed for the duration of [`block_on`].
    static RT: RefCell<Option<Rt>> = const { RefCell::new(None) };
}

struct Rt {
    reactor: Reactor<EpollDriver>,
    /// token → the waker to fire when the reactor reports that token ready. The HashTrinity demux.
    wakers: Table<usize, Waker>,
    next_token: usize,
}

/// Wake flag for the single root task — built via `std::task::Wake` (`Arc`-based), so no manual
/// `RawWaker` vtable and no `unsafe`.
struct RootWake {
    woken: AtomicBool,
}

impl Wake for RootWake {
    fn wake(self: Arc<Self>) {
        self.woken.store(true, Ordering::Release);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(true, Ordering::Release);
    }
}

/// Drive `fut` to completion on this thread, parking on `epoll` between polls. Single-task; enough to
/// run a leaf async I/O future through the reactor.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    RT.with(|r| {
        let mut slot = r.borrow_mut();
        assert!(slot.is_none(), "block_on cannot be nested on one thread");
        *slot = Some(Rt {
            reactor: Reactor::new(EpollDriver::new(64).expect("epoll_create1")),
            wakers: Table::new(),
            next_token: 1,
        });
    });

    let root = Arc::new(RootWake {
        woken: AtomicBool::new(true),
    });
    let waker = Waker::from(root.clone());
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(fut);

    let out = loop {
        // Poll the task whenever it may have progressed (initially, and after any wake).
        if root.woken.swap(false, Ordering::AcqRel) {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                break v;
            }
        }
        // Park on the reactor until a registered fd is ready, then fire that token's waker. The
        // blocking `epoll_wait` holds the RT borrow, but this thread runs nothing else meanwhile.
        let mut events = Events::new();
        let n = RT
            .with(|r| {
                r.borrow_mut()
                    .as_mut()
                    .unwrap()
                    .reactor
                    .poll(&mut events, None)
            })
            .expect("epoll_wait");
        if n == 0 {
            continue;
        }
        RT.with(|r| {
            let rt = r.borrow();
            let rt = rt.as_ref().unwrap();
            for (tok, _) in events.iter() {
                if let Some(w) = rt.wakers.get(&tok.0) {
                    w.wake_by_ref();
                }
            }
        });
    };

    RT.with(|r| *r.borrow_mut() = None);
    out
}

/// An fd registered with the running reactor — the async I/O source (`tokio`'s `AsyncFd`, `smol`'s
/// `Async`). `readable`/`writable` are futures that complete when the reactor reports the fd. The fd
/// is held as `AsFd`, so no `unsafe` and no raw-fd handling here.
pub struct Async<T: AsFd> {
    inner: T,
    token: usize,
}

impl<T: AsFd> Async<T> {
    /// Wrap `inner`, allocating a token from the running runtime. Call inside [`block_on`].
    pub fn new(inner: T) -> Self {
        let token = RT.with(|r| {
            let mut rt = r.borrow_mut();
            let rt = rt.as_mut().expect("Async::new outside block_on");
            let t = rt.next_token;
            rt.next_token += 1;
            t
        });
        Async { inner, token }
    }

    /// The wrapped source (to run the actual I/O once ready).
    pub fn get_ref(&self) -> &T {
        &self.inner
    }

    /// Complete when the fd is readable.
    pub fn readable(&self) -> Ready<'_, T> {
        Ready {
            af: self,
            interest: Interest::READABLE,
            registered: false,
        }
    }

    /// Complete when the fd is writable.
    pub fn writable(&self) -> Ready<'_, T> {
        Ready {
            af: self,
            interest: Interest::WRITABLE,
            registered: false,
        }
    }
}

/// The future from [`Async::readable`]/[`Async::writable`]: register the fd once (→ `Pending`),
/// complete once the reactor wakes it. One-shot — a re-arm is a fresh `readable()`/`writable()`.
pub struct Ready<'a, T: AsFd> {
    af: &'a Async<T>,
    interest: Interest,
    registered: bool,
}

impl<T: AsFd> Future for Ready<'_, T> {
    type Output = io::Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let token = self.af.token;
        if self.registered {
            // Woken by the reactor ⇒ ready. Deregister (one-shot) and drop the waker.
            return Poll::Ready(RT.with(|r| {
                let mut rt = r.borrow_mut();
                let rt = rt.as_mut().unwrap();
                let _ = rt.reactor.deregister(Token(token), self.af.inner.as_fd());
                rt.wakers.remove(&token);
                Ok(())
            }));
        }
        let interest = self.interest;
        let waker = cx.waker().clone();
        let res = RT.with(|r| {
            let mut rt = r.borrow_mut();
            let rt = rt.as_mut().expect("readiness polled outside block_on");
            rt.wakers.put(token, waker);
            rt.reactor
                .register(Token(token), self.af.inner.as_fd(), interest)
        });
        match res {
            Ok(()) => {
                self.registered = true;
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reactor::{pipe2_cloexec, read, write};

    #[test]
    fn block_on_returns_a_ready_value_without_touching_io() {
        assert_eq!(block_on(async { 2 + 3 }), 5);
    }

    #[test]
    fn block_on_drives_an_async_pipe_read() {
        // Level-triggered: write first, so the read end is immediately readable — this still exercises
        // register → epoll_wait → demux → wake → re-poll → read, deterministically (no thread).
        let (rd, wr) = pipe2_cloexec().unwrap();
        write(wr.as_fd(), b"x").unwrap();
        let got = block_on(async {
            let ard = Async::new(rd);
            ard.readable().await.unwrap();
            let mut buf = [0u8; 8];
            let n = read(ard.get_ref().as_fd(), &mut buf).unwrap();
            (n, buf[0])
        });
        assert_eq!(got, (1, b'x'));
    }

    #[test]
    fn block_on_parks_until_another_thread_writes() {
        // Nothing is written yet, so the executor parks on `epoll_wait(None)` (blocks indefinitely);
        // a second thread writes after a delay, and the reactor wakes the task. No timeout to miss,
        // so this proves genuine park-and-wake without flakiness.
        let (rd, wr) = pipe2_cloexec().unwrap();
        let writer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            write(wr.as_fd(), b"y").unwrap();
            drop(wr); // keep the write end open until after the write
        });
        let got = block_on(async {
            let ard = Async::new(rd);
            ard.readable().await.unwrap(); // parks on epoll until the other thread writes
            let mut buf = [0u8; 8];
            let n = read(ard.get_ref().as_fd(), &mut buf).unwrap();
            (n, buf[0])
        });
        writer.join().unwrap();
        assert_eq!(got, (1, b'y'));
    }
}
