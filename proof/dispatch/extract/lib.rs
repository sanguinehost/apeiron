//! Extract the executing admission core; wrappers only call its real bodies.
#[path = "../../../tools/dispatch/core.rs"]
pub mod core;

pub fn same(a: core::Digest, b: core::Digest) -> bool {
    a.same(b)
}
pub fn start(reg: Option<core::Digest>, observed: core::Digest) -> Option<core::Session> {
    core::start(reg, observed)
}
pub fn advance(s: core::Session, o: core::Observation) -> Option<core::Session> {
    core::advance(s, o)
}
pub fn finish(s: core::Session, observed: core::Digest) -> Option<core::Permit> {
    core::finish(s, observed)
}
#[allow(clippy::too_many_arguments)]
pub fn complete(
    reg: Option<core::Digest>,
    initial: core::Digest,
    a: core::Observation,
    b: core::Observation,
    c: core::Observation,
    d: core::Observation,
    final_digest: core::Digest,
) -> Option<core::Permit> {
    core::complete(reg, initial, a, b, c, d, final_digest)
}
pub fn witness_success() -> Option<core::Permit> {
    core::witness_success()
}
pub fn witness_refusal() -> Option<core::Permit> {
    core::witness_refusal_unregistered()
}
