//! Port of pinned CTF `studies/fast_tensor_ctr.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::*,
};

const N: usize = 4;
const S: usize = 1;
const T: usize = 1;
const V: usize = 1;
const INTERNAL_TOLERANCE: f64 = 1.0e-6;

type Tensor<'c, 'r> = SymmetricTensor<'c, 'r, Arithmetic<f64>>;

fn tensor<'c, 'r>(context: &'c Context<'r>, order: usize) -> Tensor<'c, 'r> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; order];
    if order != 0 {
        mappings[0].augment_physical(&topology, 0);
    }
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(
            Distribution::new(vec![N; order], topology, mappings),
            vec![NS; order],
        ),
        Arithmetic::new(),
    )
}

fn fixture(global_key: usize, seed: u64) -> f64 {
    let mixed = (global_key as u64)
        .wrapping_add(seed)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(29);
    ((mixed % 2003) as f64 + 1.0) / 2004.0
}

fn text(indices: &[u8]) -> String {
    String::from_utf8(indices.to_vec()).unwrap()
}

fn contract<'c, 'r>(
    output: &mut Tensor<'c, 'r>,
    output_indices: &[u8],
    left: &Tensor<'c, 'r>,
    left_indices: &[u8],
    right: &Tensor<'c, 'r>,
    right_indices: &[u8],
    alpha: f64,
    beta: f64,
) {
    let output_indices = text(output_indices);
    let left_indices = text(left_indices);
    let right_indices = text(right_indices);
    output
        .contract_from(
            &output_indices,
            left,
            &left_indices,
            right,
            &right_indices,
            alpha,
            beta,
            true,
        )
        .unwrap();
}

fn chi(indices: &[u8], p_len: usize, q_len: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    if p_len + q_len > indices.len() {
        return Vec::new();
    }
    if indices.is_empty() || (p_len == 0 && q_len == 0) {
        return vec![(Vec::new(), Vec::new())];
    }
    let last = indices[indices.len() - 1];
    let prefix = &indices[..indices.len() - 1];
    let mut pairs = Vec::new();
    if q_len == 0 {
        for (mut p, q) in chi(prefix, p_len - 1, 0) {
            p.push(last);
            pairs.push((p, q));
        }
        pairs.extend(chi(prefix, p_len, 0));
    } else if p_len == 0 {
        for (p, mut q) in chi(prefix, 0, q_len - 1) {
            q.push(last);
            pairs.push((p, q));
        }
        pairs.extend(chi(prefix, 0, q_len));
    } else {
        for (mut p, q) in chi(prefix, p_len - 1, q_len) {
            p.push(last);
            pairs.push((p, q));
        }
        for (p, mut q) in chi(prefix, p_len, q_len - 1) {
            q.push(last);
            pairs.push((p, q));
        }
        pairs.extend(chi(prefix, p_len, q_len));
    }
    pairs
}

fn subsets(indices: &[u8], p_len: usize) -> Vec<Vec<u8>> {
    chi(indices, p_len, indices.len() - p_len)
        .into_iter()
        .map(|(p, _)| p)
        .collect()
}

fn choose(n: usize, k: usize) -> usize {
    (1..=k).fold(1, |value, i| value * (n + 1 - i) / i)
}

fn check_symmetry(tsr: &Tensor<'_, '_>, antisymmetric: bool) -> (bool, f64) {
    let order = tsr.distribution().distribution().shape.len();
    let labels: Vec<_> = (0..order).map(|i| b'a' + i as u8).collect();
    let labels_text = text(&labels);
    let mut permuted = labels.clone();
    let mut difference = tensor(tsr.context(), order);
    let mut maximum: f64 = 0.0;
    for i in 0..order {
        for j in i + 1..order {
            permuted[i] = labels[j];
            permuted[j] = labels[i];
            difference.sum_from(&labels_text, tsr, &labels_text, 1.0, 1.0);
            difference.sum_from(
                &labels_text,
                tsr,
                &text(&permuted),
                if antisymmetric { 1.0 } else { -1.0 },
                1.0,
            );
            let norm = difference.norm2();
            maximum = maximum.max(norm);
            if norm > INTERNAL_TOLERANCE {
                return (false, maximum);
            }
            permuted[i] = labels[i];
            permuted[j] = labels[j];
        }
    }
    (true, maximum)
}

fn run(context: &Context<'_>) -> (f64, f64, f64) {
    let idx_c: Vec<_> = (0..S + T).map(|i| b'a' + i as u8).collect();
    let idx_a: Vec<_> = (0..S + V)
        .map(|i| {
            if i < S {
                b'a' + i as u8
            } else {
                b'a' + (S + T + i - S) as u8
            }
        })
        .collect();
    let idx_b: Vec<_> = (0..T + V)
        .map(|i| {
            if i < T {
                b'a' + (S + i) as u8
            } else {
                b'a' + (S + T + i - T) as u8
            }
        })
        .collect();

    let mut a = tensor(context, S + V);
    let mut b = tensor(context, T + V);
    let mut c = tensor(context, S + T);
    let mut c_internal = tensor(context, S + T);
    let mut c_answer = tensor(context, S + T);
    let mut vector = tensor(context, 1);
    vector.transform(|key, value| *value = fixture(key, 13));
    for &index in &idx_a {
        a.sum_from(&text(&idx_a), &vector, &text(&[index]), 1.0, 1.0);
    }
    vector.transform(|key, value| *value = fixture(key, 29));
    for &index in &idx_b {
        b.sum_from(&text(&idx_b), &vector, &text(&[index]), 1.0, 1.0);
    }

    contract(&mut c_internal, &idx_c, &a, &idx_a, &b, &idx_b, 1.0, 1.0);
    for (idx_as, idx_bt) in chi(&idx_c, S, T) {
        let mut idx_c_internal = idx_as;
        idx_c_internal.extend(idx_bt);
        c_answer.sum_from(&text(&idx_c), &c_internal, &text(&idx_c_internal), 1.0, 1.0);
    }
    let (_c_is_symmetric, c_symmetry_norm) = check_symmetry(&c_answer, false);

    let idx_z: Vec<_> = (0..S + V + T).map(|i| b'a' + i as u8).collect();
    let mut z_a_operands = tensor(context, S + V + T);
    let mut z_b_operands = tensor(context, S + V + T);
    let mut z_products = tensor(context, S + V + T);
    for indices in subsets(&idx_z, S + V) {
        z_a_operands.sum_from(&text(&idx_z), &a, &text(&indices), 1.0, 1.0);
    }
    for indices in subsets(&idx_z, T + V) {
        z_b_operands.sum_from(&text(&idx_z), &b, &text(&indices), 1.0, 1.0);
    }
    contract(
        &mut z_products,
        &idx_z,
        &z_a_operands,
        &idx_z,
        &z_b_operands,
        &idx_z,
        1.0,
        0.0,
    );
    let (_z_is_symmetric, z_symmetry_norm) = check_symmetry(&z_products, false);
    c.sum_from(&text(&idx_c), &z_products, &text(&idx_z), 1.0, 1.0);

    let mut correction_v = tensor(context, S + T);
    for r in 0..V {
        for p in V.saturating_sub(T + r)..=V - r {
            for q in V.saturating_sub(S + r)..=V - p - r {
                let prefactor = (choose(V, r)
                    * choose(V - r, p)
                    * choose(V - p - r, q)
                    * N.pow((V - p - q - r) as u32)) as f64;
                let idx_kr: Vec<_> = (0..r).map(|i| b'a' + (S + T + i) as u8).collect();
                let idx_kp: Vec<_> = (0..p).map(|i| b'a' + (S + T + r + i) as u8).collect();
                let idx_kq: Vec<_> = (0..q).map(|i| b'a' + (S + T + r + p + i) as u8).collect();
                let mut idx_va = idx_c.clone();
                idx_va.extend(&idx_kr);
                let mut va_operands = tensor(context, S + T + r);
                for selected in subsets(&idx_c, S + V - p - r) {
                    let mut idx_vaa = selected;
                    idx_vaa.extend(&idx_kr);
                    idx_vaa.extend(&idx_kp);
                    va_operands.sum_from(&text(&idx_va), &a, &text(&idx_vaa), 1.0, 1.0);
                }

                let mut idx_vb = idx_c.clone();
                idx_vb.extend(&idx_kr);
                let mut vb_operands = tensor(context, S + T + r);
                for selected in subsets(&idx_c, T + V - q - r) {
                    let mut idx_vbb = selected;
                    idx_vbb.extend(&idx_kr);
                    idx_vbb.extend(&idx_kq);
                    vb_operands.sum_from(&text(&idx_vb), &b, &text(&idx_vbb), 1.0, 1.0);
                }
                contract(
                    &mut correction_v,
                    &idx_c,
                    &va_operands,
                    &idx_va,
                    &vb_operands,
                    &idx_vb,
                    prefactor,
                    1.0,
                );
            }
        }
    }

    let mut correction_w = tensor(context, S + T);
    for r in 1..=S.min(T) {
        let idx_kr: Vec<_> = (0..r).map(|i| b'a' + (S + T + i) as u8).collect();
        let idx_kv: Vec<_> = (0..V).map(|i| b'a' + (S + T + r + i) as u8).collect();
        let mut u = tensor(context, S + T - r);
        let mut idx_u = idx_kr.clone();
        idx_u.extend(&idx_c[..S + T - 2 * r]);
        for (idx_j, idx_l) in chi(&idx_c[..S + T - 2 * r], S - r, T - r) {
            let mut idx_ua = idx_kr.clone();
            idx_ua.extend(&idx_kv);
            idx_ua.extend(idx_j);
            let mut idx_ub = idx_kr.clone();
            idx_ub.extend(&idx_kv);
            idx_ub.extend(idx_l);
            contract(&mut u, &idx_u, &a, &idx_ua, &b, &idx_ub, 1.0, 1.0);
        }
        for (idx_r, idx_h) in chi(&idx_c, r, S + T - 2 * r) {
            let mut source_indices = idx_r;
            source_indices.extend(idx_h);
            correction_w.sum_from(&text(&idx_c), &u, &text(&source_indices), 1.0, 1.0);
        }
    }

    c.sum_from(&text(&idx_c), &correction_v, &text(&idx_c), -1.0, 1.0);
    c.sum_from(&text(&idx_c), &correction_w, &text(&idx_c), -1.0, 1.0);
    c.sum_from(&text(&idx_c), &c_answer, &text(&idx_c), -1.0, 1.0);
    let final_norm = c.norm2();
    assert!(
        final_norm.is_finite() && final_norm <= 1.0e-3,
        "source fast_tensor_ctr norm={final_norm}"
    );
    (c_symmetry_norm, z_symmetry_norm, final_norm)
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let world_metrics = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let parity_metrics = run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_fast_tensor_ctr: n=4,s=t=v=1; C_ans_sym world={:e} parity={:e}; Z_mults_sym world={:e} parity={:e}; final world={:e} parity={:e}; final<=1e-3",
            world_metrics.0,
            parity_metrics.0,
            world_metrics.1,
            parity_metrics.1,
            world_metrics.2,
            parity_metrics.2,
        );
    }
    world.close();
    runtime.finalize();
}
