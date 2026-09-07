#![cfg(feature="native-linalg")]
use ctf::{algebra::{Arithmetic,Complex,Monoid,Semiring}, cost::Models,
    linalg::{Gemm,GemmKernel,Native,Transpose},partial_fold::{self,Outcome},
    partial_fold_kernel,symmetry::Symmetry::NS};

fn exercise<T:Clone+PartialEq>(value:impl Fn(i32,i32)->T,close:impl Fn(&T,&T)->bool)
where Arithmetic<T>:Semiring<Element=T>,Native:GemmKernel<T>{
    let algebra=Arithmetic::<T>::new();let alpha=value(2,1);let beta=value(1,-1);
    let (m,n,k)=(2,3,2);
    for ta in [false,true]{for tb in [false,true]{
        let lda=if ta{k+1}else{m+1};let ldb=if tb{n+1}else{k+1};let ldc=m+1;
        let mut a=vec![value(-9,0);lda*if ta{m}else{k}];
        let mut b=vec![value(-9,0);ldb*if tb{k}else{n}];
        for p in 0..k{for i in 0..m{a[if ta{p+lda*i}else{i+lda*p}]=value((i+2*p+1)as i32,1);}}
        for j in 0..n{for p in 0..k{b[if tb{j+ldb*p}else{p+ldb*j}]=value((p+j+1)as i32,-1);}}
        let mut c=vec![value(3,2);ldc*n];
        Native::gemm(Gemm{trans_a:if ta{Transpose::Yes}else{Transpose::No},
            trans_b:if tb{Transpose::Yes}else{Transpose::No},m,n,k,alpha:alpha.clone(),
            a:&a,lda,b:&b,ldb,beta:beta.clone(),c:&mut c,ldc});
        for j in 0..n{for i in 0..m{
            let mut sum=algebra.zero();for p in 0..k{
                sum=algebra.add(&sum,&algebra.multiply(&value((i+2*p+1)as i32,1),&value((p+j+1)as i32,-1)));
            }
            let expected=algebra.add(&algebra.multiply(&alpha,&sum),&algebra.multiply(&beta,&value(3,2)));
            assert!(close(&c[i+ldc*j],&expected),"typed padded GEMM");
        }assert!(c[m+ldc*j]==value(3,2),"padding untouched");}
    }}
    let mut c=vec![value(3,2);m*n];
    Native::gemm(Gemm{trans_a:Transpose::No,trans_b:Transpose::No,m,n,k:0,
        alpha:alpha.clone(),a:&[],lda:m,b:&[],ldb:1,beta:beta.clone(),c:&mut c,ldc:m});
    for x in c{assert!(close(&x,&algebra.multiply(&beta,&value(3,2))));}

    let shapes:[&[usize];3]=[&[2,2,3],&[3,2],&[2,2]];
    let links=[&[NS,NS,NS][..],&[NS,NS][..],&[NS,NS][..]];
    let Outcome::Selected(plan)=partial_fold::select(shapes,links,["xik","kj","ij"],&Models::upstream(1),[1;3]).unwrap()else{panic!("partial fold")};
    let a:Vec<_>=(0..12).map(|key|value(key+1,1)).collect();
    let b:Vec<_>=(0..6).map(|key|value(key+1,-1)).collect();let mut c=vec![value(3,2);4];
    partial_fold_kernel::execute::<T,Native>(&plan,shapes,links,["xik","kj","ij"],&a,&b,&mut c,alpha.clone(),beta.clone());
    for j in 0..2{for i in 0..2{let mut sum=algebra.zero();
        for p in 0..3{for x in 0..2{sum=algebra.add(&sum,&algebra.multiply(&a[x+2*i+4*p],&b[p+3*j]));}}
        let expected=algebra.add(&algebra.multiply(&alpha,&sum),&algebra.multiply(&beta,&value(3,2)));
        assert!(close(&c[i+2*j],&expected),"typed partial folded residuals");
    }}
}
#[test]fn four_native_scalar_kernels(){
    exercise(|r,_|r as f32,|a,b|a.is_finite()&&(a-b).abs()<1e-6);
    exercise(|r,_|r as f64,|a,b|a.is_finite()&&(a-b).abs()<1e-6);
    exercise(|r,i|Complex::new(r as f32,i as f32),|a,b|a.re.is_finite()&&a.im.is_finite()&&(a.re-b.re).abs()<1e-6&&(a.im-b.im).abs()<1e-6);
    exercise(|r,i|Complex::new(r as f64,i as f64),|a,b|a.re.is_finite()&&a.im.is_finite()&&(a.re-b.re).abs()<1e-6&&(a.im-b.im).abs()<1e-6);
}
