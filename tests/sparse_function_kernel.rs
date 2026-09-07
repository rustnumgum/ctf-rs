use ctf::{algebra::Arithmetic,sparse_formats::Coo,sparse_function_kernel::{csr_dense,csr_sparse}};
#[test]
fn folded_function_preserves_sparse_structure_and_dense_zeros(){
    let a=Coo::new(2,2,vec![(1,1,0_i64),(2,2,2)]).to_csr();
    let b=Coo::new(2,2,vec![(1,1,0_i64),(2,2,0)]).to_csr();
    let mut dense=[3;4];let mut sparse=[3;4];
    csr_dense(&Arithmetic::<i64>::new(),&a,2,&[0,1,2,0],&mut dense,&1,&3,|a,b|a+b+1);
    csr_sparse(&Arithmetic::<i64>::new(),&a,&b,&mut sparse,&1,&3,|a,b|a+b+1);
    assert_eq!(dense,[10,13,12,12]);assert_eq!(sparse,[10,9,9,12]);
}
