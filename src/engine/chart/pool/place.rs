//! Placement of a second dispatching thread beside the pool.
//!
//! The pool pins its caller to the first performance core and its first performance worker to the
//! second. A thread spawned from the caller inherits the caller's mask, so two dispatchers share one
//! core. [`pin_beside_worker`] moves such a thread to the hardware-thread sibling of the first
//! performance worker's core: the worker's spin keeps that core out of its idle states, and the
//! caller's core stays the caller's.

use std::sync::OnceLock;

/// Pins the calling thread to the SMT sibling of the first performance worker's core and marks it
/// pinned, so its pool dispatches keep that placement; with no performance list, fewer than two
/// performance cores, or no sibling, the mask is left as inherited.
///
/// The ADMM side-B thread of `learn::nanoquant` (blk.23 `ffn_down`, r 768, 100 iterations) took the
/// fit from 1.788 to 1.608 s on the i9-14900KF (sibling CPU 3), means of 3 interleaved runs, factor
/// byte-identical; the inherited mask (CPU 0) and CPU 1 measured 12.92 and 12.60 ms per iteration
/// against 11.50 on CPU 3.
///
/// # Complexity
///
/// `O(c)` sysfs reads for `c` performance CPUs on the first call of the process, `O(1)` after.
pub(crate) fn pin_beside_worker() {
    static CPU: OnceLock<Option<usize>> = OnceLock::new();
    let cpu = *CPU.get_or_init(|| {
        beside_worker(&super::performance_cpus(), |c| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
            ))
            .ok()
        })
    });
    if let Some(cpu) = cpu {
        super::CALLER_PINNED.with(|d| d.set(true));
        super::pin_current_thread(cpu);
    }
}

/// The sibling of the first performance worker's core (`p_cpus[1]`, see `Pool::with_classes`) in
/// the `thread_siblings_list` that `siblings` reads for a CPU.
///
/// # Complexity
///
/// `O(s)` in the length of one sibling list.
fn beside_worker(p_cpus: &[usize], siblings: impl Fn(usize) -> Option<String>) -> Option<usize> {
    let worker = *p_cpus.get(1)?;
    let list = siblings(worker)?;
    let mut out = None;
    for part in list.trim().split(',') {
        let mut ends = part.split('-').map(|v| v.trim().parse::<usize>());
        let Some(Ok(a)) = ends.next() else { continue };
        let b = match ends.next() {
            Some(Ok(b)) => b,
            _ => a,
        };
        out = out.or((a..=b).find(|&c| c != worker));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::beside_worker;

    /// The i9-14900KF layout: performance CPUs `0, 2, …, 14` with siblings `2k-2k+1`; efficiency
    /// CPUs single-threaded.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the shape of the topology callback"
    )]
    fn host(c: usize) -> Option<String> {
        Some(if c < 16 {
            format!("{}-{}", c & !1, c | 1)
        } else {
            c.to_string()
        })
    }

    #[test]
    fn beside_worker_takes_the_first_workers_sibling() {
        let p: Vec<usize> = (0..16).step_by(2).collect();
        assert_eq!(beside_worker(&p, host), Some(3));
        assert_eq!(beside_worker(&p, |_| Some("2,3".into())), Some(3));
        assert_eq!(beside_worker(&[16, 17], host), None, "no sibling");
        assert_eq!(beside_worker(&[0], host), None, "no worker core");
        assert_eq!(beside_worker(&p, |_| None), None, "no topology");
    }
}
