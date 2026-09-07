use ctf::{fold_layout::{FoldLayout,Direction},symmetry::Symmetry::{NS,SY,AS}};
#[test]
fn partial_group_transpose_preserves_virtual_block_boundaries(){
    let layout=FoldLayout::new(&[2,3],&[NS,NS],&[0,1],&[1]);
    let input:Vec<i64>=(0..12).collect();
    let packed=layout.transpose(&input,2,Direction::Forward);
    assert_eq!(packed,[0,2,4,1,3,5,6,8,10,7,9,11]);
    assert_eq!(layout.transpose(&packed,2,Direction::Backward),input);
}
#[test]
fn compressed_group_transpose_does_not_expand_symmetry(){
    let mut layout=FoldLayout::new(&[2,2,3,3,2],&[SY,NS,AS,NS,NS],&[0,1,2,3,4],&[2,3,4]);
    assert_eq!(layout.group_lengths,[3,3,2]);
    layout.permute_folded(&[1,0]);assert_eq!(layout.inner_ordering,[2,1,0]);
    let input:Vec<_>=(0..36).map(|i|format!("v{i}")).collect();
    let packed=layout.transpose(&input,2,Direction::Forward);assert_eq!(packed.len(),36);
    for virtual_block in 0..2{for a in 0..3{for b in 0..3{for c in 0..2{
        assert_eq!(packed[virtual_block*18+c+2*(b+3*a)],input[virtual_block*18+a+3*(b+3*c)]);
    }}}}
    assert_eq!(layout.transpose(&packed,2,Direction::Backward),input);
}
#[test]
fn scalar_and_zero_local_packed_blocks(){
    let scalar=FoldLayout::new(&[],&[],&[],&[]);assert_eq!(scalar.transpose(&[7],1,Direction::Forward),[7]);
    let empty=FoldLayout::new(&[0,2],&[NS,NS],&[0,1],&[1]);
    assert!(empty.transpose::<i64>(&[],3,Direction::Forward).is_empty());
}
