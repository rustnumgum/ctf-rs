use ctf::{fold_layout::FoldLayout,symmetry::Symmetry::{NS,SY,AS,SH}};
#[test]
fn packed_groups_selected_prefix_and_source_fold_index_positions(){
    // i,j form one SY group, k,l an AS group, m is unfolded, n,o hollow.
    let mut layout=FoldLayout::new(&[3,3,4,4,5,3,3],&[SY,NS,AS,NS,NS,SH,NS],
        &[0,1,2,3,4,5,6],&[0,1,5,6]);
    assert_eq!(layout.group_lengths,[6,10,5,6]);assert_eq!(layout.folded_shape,[6,6]);
    assert_eq!(layout.folded_indices,[1,3]);assert_eq!(layout.inner_ordering,[0,3,1,2]);
    layout.permute_folded(&[1,0]);assert_eq!(layout.inner_ordering,[3,0,1,2]);
    assert_eq!(layout.folded_shape,[6,6]);
}
#[test]
fn partial_ns_and_empty_scalar_metadata(){
    let layout=FoldLayout::new(&[2,3,4],&[NS,NS,NS],&[2,0,1],&[0,1]);
    assert_eq!(layout.group_lengths,[2,3,4]);assert_eq!(layout.folded_shape,[3,4]);
    assert_eq!(layout.folded_indices,[0,1]);assert_eq!(layout.inner_ordering,[1,2,0]);
    let scalar=FoldLayout::new(&[],&[],&[],&[]);assert!(scalar.group_lengths.is_empty());
    assert!(scalar.folded_shape.is_empty());assert!(scalar.inner_ordering.is_empty());
}
