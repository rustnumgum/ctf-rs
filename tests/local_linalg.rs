//! Local-kernel tests. Distributed ScaLAPACK validation remains a separate gate.
#![cfg(feature = "native-linalg")]
use ctf::linalg::{Gemm, GemmKernel, LocalKernels, Native, Transpose};

fn product(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.; m * n];
    Native::gemm(Gemm {
        trans_a: Transpose::No,
        trans_b: Transpose::No,
        m,
        n,
        k,
        alpha: 1.,
        a,
        lda: m.max(1),
        b,
        ldb: k.max(1),
        beta: 0.,
        c: &mut c,
        ldc: m.max(1),
    });
    c
}
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|x| x * x).sum::<f64>().sqrt()
}
fn check(q: &str, value: f64, bound: f64) {
    assert!(
        value.is_finite() && value <= bound,
        "{q}: {value} > {bound}"
    );
    println!("DIGIT / PASS {q}: delta={value:e}, bound={bound:e}");
}
fn orthogonality(rows: usize, cols: usize, q: &[f64]) -> f64 {
    let mut gram = vec![0.; cols * cols];
    Native::gemm(Gemm {
        trans_a: Transpose::Yes,
        trans_b: Transpose::No,
        m: cols,
        n: cols,
        k: rows,
        alpha: 1.,
        a: q,
        lda: rows,
        b: q,
        ldb: rows,
        beta: 0.,
        c: &mut gram,
        ldc: cols,
    });
    norm(
        gram.iter()
            .enumerate()
            .map(|(i, &x)| x - if i / cols == i % cols { 1. } else { 0. }),
    )
}

#[test]
fn gemm_integer_valued() {
    assert_eq!(
        product(
            2,
            2,
            3,
            &[1., 2., 3., 4., 5., 6.],
            &[7., 8., 9., 10., 11., 12.]
        ),
        vec![76., 100., 103., 136.]
    );
}

#[test]
fn qr_reconstruction() {
    // Frobenius bounds from upstream scalapack_tests/qr.cxx, applied here to the local interface.
    let (m, n) = (5, 3);
    let a: Vec<f64> = (0..m * n)
        .map(|i| ((i * 7 + 3) % 17) as f64 / 17.)
        .collect();
    let (q, r) = Native::qr(m, n, &a).unwrap();
    check(
        "local QR orthogonality",
        orthogonality(m, n, &q),
        (m * n) as f64 * 1e-6,
    );
    let reconstructed = product(m, n, n, &q, &r);
    check(
        "local QR reconstruction",
        norm(a.iter().zip(reconstructed).map(|(&x, y)| x - y)),
        (m * n * n) as f64 * 1e-6,
    );
}

#[test]
fn svd_reconstruction() {
    // Frobenius bounds from upstream scalapack_tests/svd.cxx.
    let (m, n) = (5, 3);
    let a: Vec<f64> = (0..m * n)
        .map(|i| ((i * 11 + 5) % 23) as f64 / 23.)
        .collect();
    let svd = Native::svd(m, n, &a).unwrap();
    check(
        "local SVD U orthogonality",
        orthogonality(m, n, &svd.u),
        (m * n) as f64 * 1e-6,
    );
    let v: Vec<_> = (0..n * n).map(|i| svd.vt[(i % n) * n + i / n]).collect();
    check(
        "local SVD V orthogonality",
        orthogonality(n, n, &v),
        (m * n) as f64 * 1e-6,
    );
    let mut us = svd.u;
    for j in 0..n {
        for i in 0..m {
            us[i + j * m] *= svd.values[j];
        }
    }
    let reconstructed = product(m, n, n, &us, &svd.vt);
    check(
        "local SVD reconstruction",
        norm(a.iter().zip(reconstructed).map(|(&x, y)| x - y)),
        (m * n * n) as f64 * 1e-6,
    );
}

#[test]
fn eigh_reconstruction() {
    // Frobenius bounds from upstream scalapack_tests/eigh.cxx.
    let n = 4;
    let a: Vec<f64> = (0..n * n)
        .map(|p| {
            let i = p % n;
            let j = p / n;
            (i + j + 1) as f64 + if i == j { 5. } else { 0. }
        })
        .collect();
    let (vectors, values) = Native::eigh(n, &a).unwrap();
    check(
        "local eigh orthogonality",
        orthogonality(n, n, &vectors),
        (n * n) as f64 * 1e-6,
    );
    let mut scaled = vectors.clone();
    for j in 0..n {
        for i in 0..n {
            scaled[i + j * n] *= values[j];
        }
    }
    let transpose: Vec<_> = (0..n * n).map(|i| vectors[(i % n) * n + i / n]).collect();
    let reconstructed = product(n, n, n, &scaled, &transpose);
    check(
        "local eigh reconstruction",
        norm(a.iter().zip(reconstructed).map(|(&x, y)| x - y)),
        (n * n) as f64 * 1e-6,
    );
}
