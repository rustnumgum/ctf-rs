// Source matrix.cxx svd_rand uses plain transpose even for complex scalars.
use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring},context::{Context,Runtime},
    linalg::Native,mapping::Distribution,tensor::Tensor};

macro_rules! check_reconstruction {
    ($expected:expr,$u:expr,$s:expr,$vt:expr,$make:ident,$algebra:ident,$norm2:ident,$context:ident,$grid:ident,$m:ident,$n:ident,$rank:ident,$label:expr) => {{
        let expected=$expected;
        let mut u=$u;
        let s=$s;
        let vt=$vt;
        let singular=s.read(&(0..$rank).collect::<Vec<_>>());
        u.transform(|key,x|*x=$algebra.multiply(x,&singular[key/$m]));
        let mut reconstructed=$make($m,$n);
        reconstructed.gemm_2d::<Native>(&u,&vt,$grid,$algebra.one(),$algebra.zero());
        let mut error=[0.];
        for((key,want),(other,got))in expected.local_pairs().into_iter().zip(reconstructed.local_pairs()){
            assert_eq!(key,other);assert!($norm2(got).is_finite());
            error[0]+=$norm2($algebra.add(&want,&$algebra.negate(&got)));
        }
        $context.sum_f64(&mut error);
        assert!(error[0].sqrt()<=($m*$n*$n)as f64*1e-6,"randomized residual {} ({})",error[0].sqrt(),$label);
    }};
}

macro_rules! scalar_case {
    ($name:ident,$t:ty,$value:expr,$norm2:expr) => {
        fn $name(context:&Context<'_>) {
            let value=$value;let norm2=$norm2;let algebra=Arithmetic::<$t>::new();
            let grid=if context.size()==4{[2,2]}else{[context.size(),1]};
            let make=|m,n|Tensor::new(context,Distribution::cyclic(vec![m,n],context.size()),algebra.clone());
            let(m,n,rank,width)=(5,4,2,3);
            for sparse in [false,true] {
                for supplied in [false,true] {
                    if !sparse {
                        let mut a=make(m,n);
                        a.transform(|key,x|{
                            let(i,j)=(key%m,key/m);
                            *x=if supplied{value(((key*17+11)%31)as f64/31.,(key%3)as f64/7.)}
                            else{value((1+i+j+2*i*j)as f64/10.,0.)};
                        });
                        // The automatic case is a rank-two real matrix for all scalar types.
                        let mut guess=make(m,width);
                        guess.transform(|key,x|*x=value(((key*7+3)%19)as f64/19.,(key%4)as f64/11.));
                        let(u,s,vt)=a.svd_randomized(grid,rank,1,1,19,if supplied{Some(&mut guess)}else{None}).unwrap();
                        assert_eq!(u.distribution().shape,vec![m,rank]);
                        assert_eq!(s.distribution().shape,vec![rank]);assert_eq!(vt.distribution().shape,vec![rank,n]);
                        let expected=if supplied {
                            assert_eq!(guess.distribution().shape,vec![m,width]);
                            let q=guess.slice(&[0..m,0..rank]);let qt=q.permute_axes(&[1,0]);
                            let mut projected=make(rank,n);
                            projected.gemm_2d::<Native>(&qt,&a,grid,algebra.one(),algebra.zero());
                            let mut expected=make(m,n);
                            expected.gemm_2d::<Native>(&q,&projected,grid,algebra.one(),algebra.zero());
                            expected
                        }else{a.clone()};
                        // Only the singular-value vector is read; matrix factors stay distributed.
                        check_reconstruction!(expected,u,s,vt,make,algebra,norm2,context,grid,m,n,rank,
                            if supplied{"dense supplied"}else{"dense automatic"});
                    } else {
                        // Keep one entire row absent.  With four cyclic ranks this
                        // gives rank 1 an empty local sparse shard while the
                        // remaining polynomial matrix still has rank two.
                        let mut dense_source=make(m,n);
                        dense_source.transform(|key,x|{
                            let(i,j)=(key%m,key/m);
                            *x=if i==1{algebra.zero()}
                            else if supplied{value(((key*17+11)%31)as f64/31.,(key%3)as f64/7.)}
                            else{value((1+i+j+2*i*j)as f64/10.,0.)};
                        });
                        let sparse_source=dense_source.clone().into_sparse(|x|*x!=algebra.zero());
                        assert!(sparse_source.local_nnz()<m*n);
                        if context.size()==4&&context.rank()==1{assert!(sparse_source.local_pairs().is_empty());}
                        let mut guess=make(m,width);
                        guess.transform(|key,x|*x=value(((key*7+3)%19)as f64/19.,(key%4)as f64/11.));
                        if supplied {
                            let mut zero_guess=guess.clone();
                            let before=zero_guess.local_pairs();
                            let original=zero_guess.distribution().clone();
                            let(zu,zs,zvt)=sparse_source.svd_randomized(grid,rank,0,1,19,Some(&mut zero_guess)).unwrap();
                            assert_eq!(zero_guess.distribution(),&original);
                            assert_eq!(zero_guess.local_pairs(),before);
                            let q=zero_guess.slice(&[0..m,0..rank]);
                            let qt=q.permute_axes(&[1,0]);
                            let mut projected=make(rank,n);
                            projected.gemm_2d::<Native>(&qt,&dense_source,grid,algebra.one(),algebra.zero());
                            let mut expected=make(m,n);
                            expected.gemm_2d::<Native>(&q,&projected,grid,algebra.one(),algebra.zero());
                            check_reconstruction!(expected,zu,zs,zvt,make,algebra,norm2,context,grid,m,n,rank,"sparse zero iterations");
                        }
                        let(u,s,vt)=sparse_source.svd_randomized(grid,rank,1,1,19,if supplied{Some(&mut guess)}else{None}).unwrap();
                        assert_eq!(u.distribution().shape,vec![m,rank]);
                        assert_eq!(s.distribution().shape,vec![rank]);assert_eq!(vt.distribution().shape,vec![rank,n]);
                        let expected=if supplied {
                            assert_eq!(guess.distribution().shape,vec![m,width]);
                            // Source svd_rand uses plain transpose for this
                            // projection, including for complex scalars.
                            let q=guess.slice(&[0..m,0..rank]);let qt=q.permute_axes(&[1,0]);
                            let mut projected=make(rank,n);
                            projected.gemm_2d::<Native>(&qt,&dense_source,grid,algebra.one(),algebra.zero());
                            let mut expected=make(m,n);
                            expected.gemm_2d::<Native>(&q,&projected,grid,algebra.one(),algebra.zero());
                            expected
                        }else{dense_source.clone()};
                        check_reconstruction!(expected,u,s,vt,make,algebra,norm2,context,grid,m,n,rank,
                            if supplied{"sparse supplied"}else{"sparse automatic"});
                    }
                }
            }
        }
    }
}
scalar_case!(real32,f32,|r:f64,_:f64|r as f32,|x:f32|(x as f64).powi(2));
scalar_case!(real64,f64,|r:f64,_:f64|r,|x:f64|x*x);
scalar_case!(complex32,Complex<f32>,|r:f64,i:f64|Complex::new(r as f32,i as f32),|x:Complex<f32>|(x.re as f64).powi(2)+(x.im as f64).powi(2));
scalar_case!(complex64,Complex<f64>,|r:f64,i:f64|Complex::new(r,i),|x:Complex<f64>|x.norm_squared());
fn run(c:&Context<'_>){real32(c);real64(c);complex32(c);complex64(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS typed_randomized_svd: four scalar types, dense+sparse randomized SVD, missing-row/empty-shard rank-two fixture, complex plain-transpose projection, oversampled and zero-iteration guess semantics, source Frobenius bound; world+parity");}
    world.close();runtime.finalize();}
