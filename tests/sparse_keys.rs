use ctf::{sparse_keys::{KeyMetadata,pin_blocks,depin_output},
    sparse_cost::{KeyPinning,Fractions},cost::Models,folding::Operand};

#[test]
fn virtual_residues_and_partial_phase_padding() {
    let metadata=KeyMetadata { shape:vec![5,3],phases:vec![4,2],
        virtual_dimensions:vec![2,1],physical_ranks:vec![1,0] };
    let global=vec![vec![(1,7_i64),(11,0)],vec![(3,-4),(13,9)]];
    let pinned=vec![vec![(0,7),(2,0)],vec![(0,-4),(2,9)]];
    assert_eq!(pin_blocks(&metadata,&global),pinned);
    assert_eq!(depin_output(&metadata,&pinned),global);
    let full=vec![vec![(0,7),(1,90),(2,0),(3,91)],vec![(0,-4),(1,92),(2,9),(3,93)]];
    assert_eq!(depin_output(&metadata,&full),global);
    let mut metadata=metadata; metadata.physical_ranks[0]=0;
    assert_eq!(depin_output(&metadata,&full),vec![vec![(0,7),(4,90),(10,0),(14,91)],vec![(2,-4),(12,9)]]);
}
#[test]
fn divisible_shapes_empty_blocks_and_scalar_keys() {
    let metadata=KeyMetadata { shape:vec![8,4],phases:vec![4,2],
        virtual_dimensions:vec![2,1],physical_ranks:vec![1,1] };
    let pinned=vec![vec![(0,1_i64),(1,2),(2,3),(3,4)],vec![(0,5),(1,6),(2,7),(3,8)]];
    let global=vec![vec![(9,1),(13,2),(25,3),(29,4)],vec![(11,5),(15,6),(27,7),(31,8)]];
    assert_eq!(depin_output(&metadata,&pinned),global);
    assert_eq!(pin_blocks(&metadata,&global),pinned);
    let empty=vec![vec![],vec![]] as Vec<Vec<(usize,i64)>>;
    assert_eq!(pin_blocks(&metadata,&empty),empty);
    assert_eq!(depin_output(&metadata,&empty),empty);
    let mut zero=metadata; zero.shape[0]=0;
    assert_eq!(pin_blocks(&zero,&empty),empty);
    assert_eq!(depin_output(&zero,&empty),empty);
    let scalar=KeyMetadata { shape:vec![],phases:vec![],virtual_dimensions:vec![],physical_ranks:vec![] };
    assert_eq!(pin_blocks(&scalar,&[vec![(0,0_i64)]]),vec![vec![(0,0)]]);
    assert_eq!(depin_output(&scalar,&[vec![(0,0_i64)]]),vec![vec![(0,0)]]);
}
#[test]
fn pin_cost_preserves_source_switch_fallthrough() {
    let mut models=Models::upstream(1);
    models.get_mut("pin_keys_mdl").set_coefficients(&[1.0,2.0]);
    let fractions=Fractions { a:0.125,b:0.25,c:0.0625 };
    for (operand,seconds,bytes) in [(Operand::A,2.25,5),(Operand::B,3.5,4),(Operand::C,3.25,1)] {
        let cost=KeyPinning { operand,dense_block_size:5,pair_sizes:[3,3,5] };
        assert_eq!(cost.fixed_time(&models,fractions),seconds);
        assert_eq!(cost.footprint(fractions),bytes);
    }
}
