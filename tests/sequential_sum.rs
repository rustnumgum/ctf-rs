use ctf::{algebra::Arithmetic,summation::{sequential,sequential_function}};

#[test]
fn transpose_and_beta() {
    let mut output = vec![10;6];
    sequential(&Arithmetic::<i64>::new(),&[2,3],"ij",&[1,2,3,4,5,6],&[3,2],"ji",&mut output,&2,&3);
    assert_eq!(output,vec![32,36,40,34,38,42]);
}
#[test]
fn reduce_and_broadcast() {
    let mut output = vec![10;6];
    sequential(&Arithmetic::<i64>::new(),&[2,2],"ik",&[1,2,3,4],&[2,3],"ij",&mut output,&2,&3);
    assert_eq!(output,vec![38,42,38,42,38,42]);
}
#[test]
fn repeated_labels() {
    let mut trace = [7];
    sequential(&Arithmetic::<i64>::new(),&[3,3],"ii",&[1,2,3,4,5,6,7,8,9],&[],"",&mut trace,&2,&3);
    assert_eq!(trace,[51]);
    let mut diagonal = [1;9];
    sequential(&Arithmetic::<i64>::new(),&[3],"i",&[2,3,4],&[3,3],"ii",&mut diagonal,&1,&0);
    assert_eq!(diagonal,[2,1,1,1,3,1,1,1,4]);
}
#[test]
fn empty_reduction_and_function_order() {
    let mut output = [3,4];
    sequential(&Arithmetic::<i64>::new(),&[2,0],"ik",&[],&[2],"i",&mut output,&1,&2);
    assert_eq!(output,[6,8]);
    sequential_function(&Arithmetic::<i64>::new(),&[2],"i",&[2,3],&[2],"i",&mut output,&2,&0,|x|x*x);
    assert_eq!(output,[16,36]);
}
