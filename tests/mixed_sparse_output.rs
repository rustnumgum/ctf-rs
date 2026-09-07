use ctf::{kernel,sparse_formats::Coo};

#[test]
fn symbolic_product_first_assignment_and_old_merge_order() {
    let a=Coo::new(2,2,vec![(1,1,2_i32),(1,2,0),(2,2,3),(2,2,-1)]).to_csr();
    let b=Coo::new(2,2,vec![(1,2,true),(2,1,true),(2,2,false)]).to_csr();
    let f=|a:&i32,b:&bool|*a as i64+i64::from(*b);
    let g=|value:i64,old:&mut i64|*old=*old*10+value;
    let c=kernel::csr_sparse(&a,&b,None,f,g);
    assert_eq!(c.to_coo().entries(),&[(1,1,1),(1,2,30),(2,1,40),(2,2,29)]);
    let old=Coo::new(2,3,vec![(1,2,7_i64),(1,2,8),(2,3,0)]).to_csr();
    let c=kernel::csr_sparse(&a,&b,Some(&old),f,g);
    assert_eq!(c.shape(),(2,3));
    assert_eq!(c.to_coo().entries(),&[(1,1,1),(1,2,110),(2,1,40),(2,2,29),(2,3,0)]);
}
#[test]
fn cancellation_keeps_structure_and_empty_product_keeps_old() {
    let a=Coo::new(1,2,vec![(1,1,1_i32),(1,2,-1)]).to_csr();
    let b=Coo::new(2,1,vec![(1,1,2_i64),(2,1,2)]).to_csr();
    let c=kernel::csr_sparse(&a,&b,None,|a,b|*a as i64*b,|value,old|*old+=value);
    assert_eq!(c.to_coo().entries(),&[(1,1,0)]);
    let a=Coo::<i32>::new(1,0,vec![]).to_csr();
    let b=Coo::<i64>::new(0,1,vec![]).to_csr();
    let old=Coo::new(1,1,vec![(1,1,7_i64)]).to_csr();
    let c=kernel::csr_sparse(&a,&b,Some(&old),|a,b|*a as i64*b,|value,old|*old+=value);
    assert_eq!(c,old);
}
