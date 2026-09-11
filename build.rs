//! Library search paths for the native BLAS, LAPACK, and ScaLAPACK links.
//!
//! Linux and Windows resolve `blas`, `lapack`, `scalapack-openmpi`, `openblas`,
//! and `scalapack` from the system linker path. macOS needs the Homebrew
//! prefix, which the linker does not search by default.
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let prefix = env::var("HOMEBREW_PREFIX")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            Command::new("brew")
                .arg("--prefix")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
        })
        .unwrap_or_else(|| PathBuf::from("/opt/homebrew"));
    for library in ["lib", "opt/openblas/lib", "opt/scalapack/lib"] {
        let path = prefix.join(library);
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
}
