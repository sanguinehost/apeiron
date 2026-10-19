//! The run glue: the `Q32.32` `grad::model` weights below the `f64` fold driver
//! ([`convert::teacher_from_grad`], [`convert::student_to_grad`]), and the token-stream input
//! ([`npy::read_npy_i4`], [`npy::chunks`], [`npy::split`]).

/// Conversions between the engine-loaded `Q32.32` [`crate::learn::grad::model::Weights`] and the
/// `f64` [`crate::learn::fold::layer::driver`] teacher/student tensors.
pub mod convert;
/// The fold-fit-to-finisher glue: [`finish::planes_from_fits`] and [`finish::tunables_from_weights`]
/// build the P-step from a [`crate::learn::fold::layer::driver::ModelResult`], and
/// [`finish::finish_model`] runs the KL steps and assembles the written student.
pub mod finish;
/// The NumPy `.npy` token reader and the calibration / in-sample / held-out chunk split.
pub mod npy;
/// The fold pack: the fitted `E8 ∪ O8` planes as one file, each plane's row grid stored as a
/// dictionary of `(lo, sp)` pairs.
pub mod pack;
/// The resumable model-level transfer run: `h0` from the calibration chunks, the crash-safe
/// checkpoint of the driver's [`crate::learn::fold::layer::driver::ModelState`], and the student
/// `GGUF` writer.
pub mod transfer;
