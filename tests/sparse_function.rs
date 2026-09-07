use ctf::{algebra::Arithmetic,sparse_sequential::sequential_function};
use std::cell::Cell;

#[test]
fn stored_zeros_are_evaluated_but_missing_entries_are_not() {
    let a=[(0,0_i64)];let b=[0,1,2,0];let mut c=[3;8];
    let calls=Cell::new(0);
    sequential_function(&Arithmetic::<i64>::new(),&[],"",&a,&[4],"i",&b,
        &[4,2],"ix",&mut c,&1,&3,|a,b|{calls.set(calls.get()+1);2*a+b+1});
    let expected:Vec<_>=(0..8).map(|key|10+b[key%4]).collect();
    assert_eq!(c.as_slice(),expected);assert_eq!(calls.get(),8);
}

#[test]
fn empty_structure_does_not_invoke_function() {
    let mut c=[3_i64,4];
    sequential_function(&Arithmetic::<i64>::new(),&[],"",&[],&[2],"i",&[1,2],
        &[2],"i",&mut c,&7,&2,|_,_|panic!("function called for missing sparse entry"));
    assert_eq!(c,[6,8]);
}

#[test]
fn pinned_all_scalar_path_uses_multiplication_instead_of_function() {
    let mut c=[5_i64];
    sequential_function(&Arithmetic::<i64>::new(),&[],"",&[(0,2)],&[],"",&[3],
        &[],"",&mut c,&4,&2,|_,_|panic!("pinned scalar path ignores custom function"));
    assert_eq!(c,[34]);
}
