use ctf::{fold_indices::{select,Eligibility},symmetry::Symmetry::{NS,SY,AS,SH},fold_layout::FoldLayout};
#[test]
fn source_ns_partial_and_sparse_gates(){
    let links=[&[NS,NS,NS][..],&[NS,NS][..],&[NS,NS][..]];
    let selected=select(["ikx","kj","ij"],links,false,false).unwrap();
    assert_eq!(selected.eligibility,Eligibility::Eligible);assert_eq!(selected.labels,[0,1,3]);
    assert_eq!(select(["ikx","kj","ij"],links,true,false).unwrap().eligibility,Eligibility::SparseNotFullyFolded);
    assert_eq!(select(["ij","ij","ij"],[&[NS,NS];3],true,false).unwrap().eligibility,Eligibility::SparseWeighIndex);
    assert_eq!(select(["ik","kj","ij"],[&[NS,NS];3],false,true).unwrap().eligibility,Eligibility::DenseCustom);
    assert_eq!(select(["ik","kj","ij"],[&[NS,NS];3],true,true).unwrap().eligibility,Eligibility::Eligible);
    assert!(matches!(select(["ii","i",""],[&[NS,NS],&[NS],&[]],false,false).unwrap().eligibility,Eligibility::RepeatedLabel{operand:ctf::folding::Operand::A,..}));
    assert_eq!(select(["","",""],[&[];3],false,false).unwrap().eligibility,Eligibility::NoFoldableLabels);
}
#[test]
fn source_link_matching_reversal_and_partial_packed_layout(){
    for kind in [SY,AS,SH]{
        let links=[kind,NS];
        let selected=select(["ab","ab",""],[&links,&links,&[]],false,false).unwrap();
        assert_eq!(selected.eligibility,Eligibility::Eligible);assert_eq!(selected.labels,[0,1]);
        let reversed=select(["ab","ba",""],[&links,&links,&[]],false,false).unwrap();
        assert_eq!(reversed.eligibility,Eligibility::NoFoldableLabels);assert!(reversed.labels.is_empty());
    }
    assert!(select(["ab","ab",""],[&[AS,NS],&[SH,NS],&[]],false,false).unwrap().labels.is_empty());
    let triple=select(["ab","ab","ab"],[&[SY,NS];3],false,false).unwrap();assert_eq!(triple.labels,[0,1]);
    let selected=select(["abx","ab",""],[&[SY,NS,NS],&[SY,NS],&[]],false,false).unwrap();
    assert_eq!(selected.labels,[0,1]);
    let layout=FoldLayout::new(&[3,3,4],&[SY,NS,NS],&[0,1,2],&selected.labels);
    assert_eq!(layout.group_lengths,[6,4]);assert_eq!(layout.folded_shape,[6]);assert_eq!(layout.folded_indices,[1]);
}
