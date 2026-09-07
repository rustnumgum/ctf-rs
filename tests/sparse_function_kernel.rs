use ctf::{algebra::Arithmetic,sparse_formats::Coo,sparse_function_kernel::{csr_dense,csr_sparse,csr_sparse_output}};
#[test]
fn folded_function_preserves_sparse_structure_and_dense_zeros(){
    let a=Coo::new(2,2,vec![(1,1,0_i64),(2,2,2)]).to_csr();
    let b=Coo::new(2,2,vec![(1,1,0_i64),(2,2,0)]).to_csr();
    let mut dense=[3;4];let mut sparse=[3;4];
    csr_dense(&Arithmetic::<i64>::new(),&a,2,&[0,1,2,0],&mut dense,&1,&3,|a,b|a+b+1);
    csr_sparse(&Arithmetic::<i64>::new(),&a,&b,&mut sparse,&1,&3,|a,b|a+b+1);
    assert_eq!(dense,[10,13,12,12]);assert_eq!(sparse,[10,9,9,12]);
}

#[test]
fn sparse_output_retains_structural_zero_and_merges_old_rows(){
    let a=Coo::new(2,2,vec![(1,1,0_i64),(1,2,1),(2,2,2)]).to_csr();
    let b=Coo::new(2,2,vec![(1,1,0_i64),(2,1,-1),(2,2,0)]).to_csr();
    let c=Coo::new(2,2,vec![(2,1,3_i64),(2,2,5)]).to_csr();
    let result=csr_sparse_output(&Arithmetic::<i64>::new(),&a,&b,&c,|a,b|a+b);
    assert_eq!(result.to_coo().entries(),vec![(1,1,0),(1,2,1),(2,1,4),(2,2,7)]);
}
