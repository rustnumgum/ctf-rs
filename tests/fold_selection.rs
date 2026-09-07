#![cfg(feature="native-linalg")]
use ctf::{cost::Models,folding::Plan,linalg::Native};
fn model(coefficient:f64)->Models{
    let mut m=Models::upstream(1);
    for name in ["non_contig_transp_mdl","shrt_contig_transp_mdl","long_contig_transp_mdl"]{
        m.get_mut(name).set_coefficients(&[0.,coefficient]);
    }m
}
fn shape(indices:&str)->Vec<usize>{indices.bytes().map(|label|match label{b'i'=>3,b'k'=>4,b'j'=>5,b'l'=>2,_=>unreachable!()}).collect()}
fn coordinates(indices:&str,mut key:usize)->[usize;256]{
    let mut result=[0;256];for(label,length)in indices.bytes().zip(shape(indices)){
        result[label as usize]=key%length;key/=length;
    }result
}
fn a(c:&[usize;256])->f64{(1+c[b'i' as usize]+2*c[b'k' as usize]+3*c[b'l' as usize])as f64}
fn b(c:&[usize;256])->f64{2.+c[b'k' as usize]as f64-c[b'j' as usize]as f64+2.*c[b'l' as usize]as f64}
fn execute(indices:[&str;3],models:&Models,copies:[usize;3],commutative:bool,permutation:usize)->Plan{
    let shapes=indices.map(shape);let p=Plan::select(shapes.each_ref().map(Vec::as_slice),indices,models,copies,commutative).unwrap();
    assert_eq!(p.permutation(),permutation);
    let aa:Vec<_>=(0..shapes[0].iter().product()).map(|key|a(&coordinates(indices[0],key))).collect();
    let bb:Vec<_>=(0..shapes[1].iter().product()).map(|key|b(&coordinates(indices[1],key))).collect();
    let mut output=vec![4.;shapes[2].iter().product()];p.execute::<Native>(&aa,&bb,&mut output,2.,3.);
    for(key,&value)in output.iter().enumerate(){let mut c=coordinates(indices[2],key);let mut sum=0.;
        for k in 0..4{c[b'k' as usize]=k;sum+=a(&c)*b(&c);}
        assert!(value.is_finite()&&(value-(2.*sum+12.)).abs()<1e-6,"permutation {permutation}, key {key}");
    }p
}
#[test]
fn source_six_transpose_layouts_execute_with_blas(){
    let m=model(1.);
    let layouts=[["ki","kj","ij"],["ik","kj","ij"],["ik","jk","ij"],
        ["ki","jk","ji"],["ki","kj","ji"],["ik","jk","ji"]];
    for(permutation,indices)in layouts.into_iter().enumerate(){
        let p=execute(indices,&m,[1;3],true,permutation);assert_eq!(p.transpose_seconds(),[0.;3]);
    }
}
#[test]
fn source_last_tie_noncommuting_limit_and_permuted_output_cost(){
    let zero=model(0.);
    execute(["lik","lkj","lij"],&zero,[1;3],true,5);
    execute(["lik","lkj","lij"],&zero,[1;3],false,2);
    let p=execute(["lik","lkj","lij"],&model(1.),[2,3,5],true,5);
    // Batch-first means all operands transpose for every choice. The source
    // doubles permuted C, which is original A for permutation5, not original C.
    assert_eq!(p.transpose_seconds(),[96.,120.,150.]);
}
