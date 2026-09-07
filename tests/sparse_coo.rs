use ctf::{algebra::Arithmetic,sparse_formats::Coo};

#[test]
fn default_generic_and_kernel_coo_paths() {
    let a=Coo::new(2,2,vec![(2,2,3_i64),(1,1,2),(2,2,-1),(1,2,0)]);
    let b=[1,2,3,4]; let mut c=[10,20,30,40];
    a.default_coomm(2,&b,&2,&3,&mut c);
    assert_eq!(c,[34,68,102,136]);
    let mut c=[10,20,30,40];
    a.coomm(&Arithmetic::<i64>::new(),2,&b,&2,&1,&mut c);
    assert_eq!(c,[14,28,42,56]);
    let mut c=[0;4];
    a.coomm_kernel(&Arithmetic::<i64>::new(),2,&b,Some(&1),&1,&mut c,
        |a,b|a+b,|value,old|*old=*old*10+value);
    assert_eq!(c,[32,51,54,73]); // stored order and explicit zero are observable
    let mut mixed=[0_i32;4];
    a.coomm_kernel(&Arithmetic::<i32>::new(),2,&[false,true,true,false],None,&1,&mut mixed,
        |a,b|*a as i32+i32::from(*b),|value,old|*old+=value);
    assert_eq!(mixed,[3,4,3,2]);
}
#[test]
fn empty_coo_scales_dense_output() {
    let a=Coo::<i64>::new(2,0,vec![]); let mut c=[2,3,4,5];
    a.default_coomm(2,&[],&7,&3,&mut c); assert_eq!(c,[6,9,12,15]);
    a.coomm(&Arithmetic::<i64>::new(),2,&[],&7,&1,&mut c); assert_eq!(c,[6,9,12,15]);
}
