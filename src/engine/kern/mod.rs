//! Kernels of the decode-dominance plan (`docs/epics/decode-dominance.md`): the exact
//! lookup-table row dot and the k-column projection. Each is tested bit for bit against
//! `matvec_fixed` on the dequantized `quant_act` activation.

pub(crate) mod bitplane;
pub(crate) mod gemm;
pub(crate) mod lut;
pub(crate) mod matmat;
pub(crate) mod pow2;
pub(crate) mod tp;
