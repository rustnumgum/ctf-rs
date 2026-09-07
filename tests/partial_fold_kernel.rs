#![cfg(feature="native-linalg")]
use ctf::{cost::Models,linalg::Native,partial_fold::{self,Descriptor,Outcome},partial_fold_kernel,
    symmetry::Symmetry::{self,NS,SY,AS,SH}};
fn plan(shapes:[&[usize];3],links:[&[Symmetry];3],indices:[&str;3])->Descriptor{
    let Outcome::Selected(d)=partial_fold::select(shapes,links,indices,&Models::upstream(1),[1;3]).unwrap()
        else{panic!("source rejected test fold")};d
}
fn close(a:f64,b:f64){assert!(a.is_finite()&&(a-b).abs()<1e-6,"{a} != {b}");}
#[test]
fn singleton_residuals_remain_outside_blas_and_beta_applies_once(){
    let shapes:[&[usize];3]=[&[2,2,3],&[3,2,2],&[2,2,2]];
    let links=[&[NS,NS,NS][..];3];let indices=["xik","kjy","ijz"];let d=plan(shapes,links,indices);
    let a:Vec<_>=(1..=12).map(|x|x as f64).collect();let b=a.clone();let mut c=vec![3.;8];
    partial_fold_kernel::execute::<Native>(&d,shapes,links,indices,&a,&b,&mut c,2.,4.);
    for z in 0..2{for j in 0..2{for i in 0..2{
        let mut sum=0.;for y in 0..2{for k in 0..3{for x in 0..2{sum+=a[x+2*(i+2*k)]*b[k+3*(j+2*y)];}}}
        close(c[i+2*(j+2*z)],2.*sum+12.);
    }}}
}
#[test]
fn shared_residual_symmetric_coordinates_use_packed_offsets(){
    let shapes:[&[usize];3]=[&[3,3,2],&[3,2,2],&[2]];
    let links:[&[Symmetry];3]=[&[SY,NS,NS],&[NS,NS,NS],&[NS]];let indices=["xyk","ykj","j"];
    let d=plan(shapes,links,indices);let a:Vec<_>=(1..=12).map(|x|x as f64).collect();let b=a.clone();let mut c=vec![3.;2];
    partial_fold_kernel::execute::<Native>(&d,shapes,links,indices,&a,&b,&mut c,2.,4.);
    for j in 0..2{let mut sum=0.;for y in 0..3{for x in 0..=y{for k in 0..2{
        let packed=x+y*(y+1)/2;sum+=a[packed+6*k]*b[y+3*(k+2*j)];
    }}}close(c[j],2.*sum+12.);}
}
#[test]
fn partial_outer_loop_preserves_folded_batched_gemm(){
    let shapes:[&[usize];3]=[&[2,2,3,2],&[3,2,2],&[2,2,2]];
    let links:[&[Symmetry];3]=[&[NS,NS,NS,NS],&[NS,NS,NS],&[NS,NS,NS]];let indices=["xikl","kjl","ijl"];
    let d=plan(shapes,links,indices);assert_eq!(d.batches,2);
    let a:Vec<_>=(1..=24).map(|x|x as f64).collect();let b:Vec<_>=(1..=12).map(|x|x as f64).collect();let mut c=vec![3.;8];
    partial_fold_kernel::execute::<Native>(&d,shapes,links,indices,&a,&b,&mut c,2.,4.);
    for l in 0..2{for j in 0..2{for i in 0..2{let mut sum=0.;for k in 0..3{for x in 0..2{
        sum+=a[x+2*(i+2*(k+3*l))]*b[k+3*(j+2*l)];
    }}close(c[i+2*(j+2*l)],2.*sum+12.);}}}
}
#[test]
fn residual_symmetric_output_and_fully_folded_compressed_pairs(){
    let shapes:[&[usize];3]=[&[2,3],&[2,3],&[2,2]];
    let links:[&[Symmetry];3]=[&[NS,NS],&[NS,NS],&[SY,NS]];let indices=["ik","jk","ij"];
    let d=plan(shapes,links,indices);let a:Vec<_>=(1..=6).map(|x|x as f64).collect();let b=a.clone();let mut c=vec![3.;3];
    partial_fold_kernel::execute::<Native>(&d,shapes,links,indices,&a,&b,&mut c,2.,4.);
    for j in 0..2{for i in 0..=j{let sum:f64=(0..3).map(|k|a[i+2*k]*b[j+2*k]).sum();close(c[i+j*(j+1)/2],2.*sum+12.);}}
    for(kind,k)in[(SY,6),(AS,6),(SH,6)]{
        let shapes:[&[usize];3]=[&[3,3,2],&[3,3,2],&[2,2]];
        let links:[&[Symmetry];3]=[&[kind,NS,NS],&[kind,NS,NS],&[NS,NS]];let indices=["abi","abj","ij"];
        let d=plan(shapes,links,indices);let mut a:Vec<_>=(1..=2*k).map(|x|x as f64).collect();
        if kind!=SY{for block in 0..2{for hole in [0,2,5]{a[block*k+hole]=0.;}}}
        let b=a.clone();let mut c=vec![f64::NAN;4];
        partial_fold_kernel::execute::<Native>(&d,shapes,links,indices,&a,&b,&mut c,2.,0.);
        // Inner canonical packed result only; full tensor symmetry multiplicity
        // and diagonal prescaling belong to the outer contraction orchestration.
        for j in 0..2{for i in 0..2{let sum:f64=(0..k).map(|g|a[g+k*i]*b[g+k*j]).sum();close(c[i+2*j],2.*sum);}}
    }
}
