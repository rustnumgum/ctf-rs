use ctf::{cost::Models,partial_fold::{self,Outcome},symmetry::Symmetry::{NS,SY,AS,SH},fold_layout::Direction};
fn models()->Models{let mut m=Models::upstream(1);
    for name in ["non_contig_transp_mdl","shrt_contig_transp_mdl","long_contig_transp_mdl"]{
        m.get_mut(name).set_coefficients(&[0.,1.]);
    }m
}
#[test]
fn residual_dimensions_participate_in_transpose_cost_and_storage_order(){
    let Outcome::Selected(d)=partial_fold::select([&[2,3,4],&[4,5],&[3,5]],
        [&[NS,NS,NS],&[NS,NS],&[NS,NS]],["xik","kj","ij"],&models(),[1;3]).unwrap()else{panic!("partial fold rejected")};
    assert_eq!(d.fold_labels,[1,2,3]);assert_eq!(d.permutation,1);
    assert_eq!(d.transpose_seconds,[24.,0.,0.]);assert_eq!((d.m,d.n,d.k,d.batches),(3,5,4,1));
    assert_eq!(d.layouts[0].group_lengths,[2,3,4]);assert_eq!(d.layouts[0].inner_ordering,[1,2,0]);
    let input:Vec<i64>=(0..48).collect();let packed=d.layouts[0].transpose(&input,2,Direction::Forward);
    for v in 0..2{for x in 0..2{for k in 0..4{for i in 0..3{
        assert_eq!(packed[v*24+i+3*k+12*x],input[v*24+x+2*(i+3*k)]);
    }}}}
    assert_eq!(d.layouts[0].transpose(&packed,2,Direction::Backward),input);
}
#[test]
fn symmetric_groups_become_packed_gemm_dimensions(){
    for(kind,k)in[(SY,6),(AS,3),(SH,3)]{
        let Outcome::Selected(d)=partial_fold::select([&[3,3,2],&[3,3,4],&[2,4]],
            [&[kind,NS,NS],&[kind,NS,NS],&[NS,NS]],["abi","abj","ij"],&models(),[1;3]).unwrap()else{panic!("symmetric fold rejected")};
        assert_eq!((d.m,d.n,d.k,d.batches),(2,4,k,1));assert_eq!(d.permutation,0);
        assert_eq!(d.transpose_seconds,[0.;3]);assert_eq!(d.layouts[0].folded_shape,[k,2]);
        assert_eq!(d.layouts[0].folded_indices,[1,2]);assert_eq!(d.layouts[1].folded_indices,[1,3]);
    }
    assert!(matches!(partial_fold::select([&[2],&[3],&[]],[&[NS],&[NS],&[]],
        ["x","y",""],&models(),[1;3]).unwrap(),Outcome::Ineligible(_)));
}
