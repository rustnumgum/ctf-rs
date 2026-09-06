pub(crate) mod mpi;
#[cfg(feature = "native-linalg")]
pub(crate) mod linalg;
#[cfg(feature = "native-linalg")]
pub(crate) mod factor;
#[cfg(feature = "native-scalapack")]
pub(crate) mod scalapack;
