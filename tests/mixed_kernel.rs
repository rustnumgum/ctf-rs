use ctf::{kernel,linalg::Transpose,sparse_formats::Coo};

#[test]
fn dense_transposes_and_mixed_accumulator_order() {
    let a=[1_i32,2,3,4];let at=[1,3,2,4];
    let b=[true,false,true,false];let bt=[true,true,false,false];
    for ta in [Transpose::No,Transpose::Yes] {for tb in [Transpose::No,Transpose::Yes] {
        let mut c=[0_i64;4];
        kernel::gemm(ta,tb,2,2,2,if matches!(ta,Transpose::No){&a}else{&at},
            if matches!(tb,Transpose::No){&b}else{&bt},&mut c,
            |a,b|*a as i64+i64::from(*b),|v,c|*c=*c*10+v);
        assert_eq!(c,[23,34,23,34]);
    }}
    let mut c=[7_i64;4];
    kernel::gemm(Transpose::No,Transpose::No,2,2,0,&[] as &[i32],&[] as &[bool],&mut c,
        |a,b|*a as i64+i64::from(*b),|v,c|*c+=v);
    assert_eq!(c,[7;4]);
}
#[test]
fn sparse_mixed_paths_keep_explicit_zeros_and_structure() {
    let a=Coo::new(2,2,vec![(2,2,3_i32),(1,1,2),(2,2,-1),(1,2,0)]).to_csr();
    let b=[false,true,true,false];let mut c=[0_i64;4];
    kernel::csr_dense(&a,2,&b,&mut c,|a,b|*a as i64+i64::from(*b),|v,c|*c=*c*10+v);
    assert_eq!(c,[21,40,30,29]);
    let b=Coo::new(2,2,vec![(1,2,true),(2,1,true),(2,2,false)]).to_csr();
    let mut c=[0_i64;4];
    kernel::csr_sparse_dense(&a,&b,&mut c,|a,b|*a as i64+i64::from(*b),|v,c|*c=*c*10+v);
    assert_eq!(c,[1,40,30,29]); // absent B(1,1) is not evaluated
    let mut c=[1_i64,99,2,99,3];
    kernel::accumulate_strided(3,&[4_i64,88,5,88,6],2,&mut c,2,|v,c|*c=*c*10+v);
    assert_eq!(c,[14,99,25,99,36]);
}
