use ctf::{
    algebra::{Group, Monoid, Semiring},
    mapping::{Mapping, Topology},
    scale_tsr::{for_each_block_mut, virtual_offsets},
    strp_tsr::{strip_diagonal, StripPlan},
    sym_seq_scl,
    symmetry::{Layout, Packed, Symmetry},
};

#[derive(Clone, Copy)]
struct Words;
impl Monoid for Words {
    type Element = String;
    fn zero(&self) -> String { String::new() }
    fn add(&self, left: &String, right: &String) -> String { format!("{left}+{right}") }
}
impl Group for Words {
    fn negate(&self, value: &String) -> String { format!("-{value}") }
}
impl Semiring for Words {
    fn one(&self) -> String { "1".into() }
    fn multiply(&self, left: &String, right: &String) -> String {
        format!("{left}>{right}")
    }
}

#[test]
fn packed_repeated_index_scales_diagonal_on_the_right_then_transforms() {
    let layout = Layout::new(vec![3, 3], vec![Symmetry::SY, Symmetry::NS]);
    assert_eq!(
        layout.coordinates().collect::<Vec<_>>(),
        vec![vec![0, 0], vec![0, 1], vec![1, 1], vec![0, 2], vec![1, 2], vec![2, 2]]
    );
    let mut values = (0..layout.len()).map(|index| index.to_string()).collect::<Vec<_>>();
    sym_seq_scl::transform(
        &Words,
        &layout,
        &[0, 0],
        &mut values,
        Some(&"R".into()),
        |value| value.push('!'),
    );
    assert_eq!(values, ["0>R!", "1", "2>R!", "3", "4", "5>R!"]);
    assert_eq!(sym_seq_scl::inverse_indices(&[2, 0, 2, 1]), [Some(1), Some(3), Some(2)]);

    let mut packed = Packed::new(layout, Words);
    packed.values_mut().clone_from_slice(&["a".into(), "b".into(), "c".into(), "d".into(), "e".into(), "f".into()]);
    packed.scale_indexed("ii", &"R".into());
    assert_eq!(packed.values(), ["a>R", "b", "c>R", "d", "e", "f>R"]);
}

#[test]
fn virtual_tuple_order_and_repeated_label_offsets_match_scl_virt() {
    // Tensor axes i,j,i have 2x4x2 blocks; only the i-diagonal is visited.
    let selected = [0, 9, 2, 11, 4, 13, 6, 15];
    assert_eq!(virtual_offsets(&[2, 4], &[0, 1, 0]), selected);
    let mut data = (0..32).collect::<Vec<i64>>();
    for_each_block_mut(&mut data, 2, &[2, 4], &[0, 1, 0], |block| {
        for value in block { *value = -*value; }
    });
    for block in 0..16 {
        for element in 0..2 {
            let original = (2 * block + element) as i64;
            let expected = if selected.contains(&block) {
                -original
            } else {
                original
            };
            assert_eq!(data[2 * block + element], expected);
        }
    }
}

#[test]
fn strip_and_restore_copy_only_the_source_hyper_rectangle() {
    // Axis zero is contiguous. Select x=2..4, every y=0..5, z=0..2.
    let plan = StripPlan::new(vec![4, 5, 4], vec![2, 1, 2], vec![1, 0, 0], 1);
    let data = (0..80).collect::<Vec<i64>>();
    let expected = (0..2)
        .flat_map(|z| (0..5).flat_map(move |y| [2 + 4 * y + 20 * z, 3 + 4 * y + 20 * z]))
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    assert_eq!(plan.stripped_len(), 20);
    assert_eq!(plan.strip(&data), expected);

    let mut restored = data.clone();
    let replacement = (100..120).collect::<Vec<i64>>();
    plan.restore(&replacement, &mut restored);
    let mut cursor = 0;
    for z in 0..4 {
        for y in 0..5 {
            for x in 0..4 {
                let offset = x + 4 * y + 20 * z;
                if x >= 2 && z < 2 {
                    assert_eq!(restored[offset], replacement[cursor]);
                    cursor += 1;
                } else {
                    assert_eq!(restored[offset], data[offset]);
                }
            }
        }
    }
    assert_eq!(cursor, replacement.len());
}

#[test]
fn strip_diagonal_uses_the_physical_rank_and_reduces_block_metadata() {
    let topology = Topology::new(vec![2]);
    let mut physical = Mapping::Unmapped;
    physical.augment_physical(&topology, 0);
    let mut virtual_mapping = Mapping::Unmapped;
    virtual_mapping.augment_virtual(2);
    let mut block_edges = vec![4, 6];
    let mut local_size = 24;
    let plan = strip_diagonal(
        &[0, 0],
        3,
        &[physical, virtual_mapping],
        &topology,
        1,
        &mut block_edges,
        &mut local_size,
    )
    .unwrap();
    assert_eq!(plan.edge_lengths(), [1, 2]);
    assert_eq!(plan.strip_dimensions(), [1, 2]);
    assert_eq!(plan.strip_indices(), [0, 1]);
    assert_eq!(block_edges, [4, 3]);
    assert_eq!(local_size, 12);
    assert_eq!(plan.strip(&[0, 1, 2, 3, 4, 5]), [3, 4, 5]);
}
