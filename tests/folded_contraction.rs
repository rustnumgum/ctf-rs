#![cfg(feature="native-linalg")]
use ctf::{contraction::{Folded,folded_f64},linalg::{Native,Transpose}};
#[test]
fn contiguous_batches() {
    let plan = Folded {m:2,n:2,k:3,batches:2,trans_a:Transpose::No,trans_b:Transpose::No,transposed_output:false};
    let mut c = [10.;8];
    folded_f64::<Native>(plan,&[1.,2.,3.,4.,5.,6.,1.,2.,3.,4.,5.,6.],
        &[7.,8.,9.,10.,11.,12.,7.,8.,9.,10.,11.,12.],&mut c,2.,3.);
    assert_eq!(c,[182.,230.,236.,302.,182.,230.,236.,302.]);
}
#[test]
fn transposed_output_operands() {
    let plan = Folded {m:2,n:2,k:3,batches:1,trans_a:Transpose::Yes,trans_b:Transpose::Yes,transposed_output:true};
    let mut c = [0.;4];
    folded_f64::<Native>(plan,&[1.,2.,3.,4.,5.,6.],&[7.,8.,9.,10.,11.,12.],&mut c,1.,0.);
    assert_eq!(c,[76.,103.,100.,136.]);
}
#[test]
fn zero_reduction() {
    let plan = Folded {m:2,n:2,k:0,batches:1,trans_a:Transpose::No,trans_b:Transpose::No,transposed_output:false};
    let mut c = [1.,2.,3.,4.];
    folded_f64::<Native>(plan,&[],&[],&mut c,1.,2.);
    assert_eq!(c,[2.,4.,6.,8.]);
}
