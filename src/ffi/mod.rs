#[cfg(feature = "native-linalg")]
pub(crate) mod linalg;
#[cfg(feature = "native-linalg")]
pub(crate) mod factor;
pub(crate) mod mpi;
#[cfg(feature = "native-scalapack")]
pub(crate) mod scalapack;
