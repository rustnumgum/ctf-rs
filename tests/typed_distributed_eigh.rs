// Pinned scalapack_tests/eigh.cxx: Frobenius residuals <= n*n*1e-6.
use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring},context::{Context,Runtime},
    linalg::Native,mapping::Distribution,tensor::Tensor};
macro_rules! scalar_case {
    ($name:ident,$t:ty,$value:expr,$conjugate:expr,$norm2:expr,$real:expr) => {
        fn $name(context:&Context<'_>) {
            let value=$value;let conjugate=$conjugate;let norm2=$norm2;let real=$real;
            let algebra=Arithmetic::<$t>::new();
            let grid=if context.size()==4{[2,2]}else{[context.size(),1]};
            for(n,degenerate)in[(5,false),(5,true),(1,false)] {
                let mut a=Tensor::new(context,Distribution::cyclic(vec![n,n],context.size()),algebra.clone());
                let input=|key:usize|{
                    let(i,j)=(key%n,key/n);
                    if degenerate {value(if i==j{(i/2)as f64-1.}else{0.},0.)}
                    else if i==j {value(i as f64/4.-0.5,0.)}
                    else {value((i+j+1)as f64/20.,(i as f64-j as f64)/30.)}
                };
                a.transform(|key,x|*x=input(key));
                let(mut vectors,values)=a.eigh().unwrap();
                assert_eq!(vectors.distribution(),a.distribution());
                for(key,x)in a.local_pairs(){assert!(x==input(key),"eigh input unchanged");}
                let eigenvalues=values.read(&(0..n).collect::<Vec<_>>());
                assert!(eigenvalues.iter().all(|&x|norm2(x).is_finite()&&x==value(real(x),0.)));
                assert!(eigenvalues.windows(2).all(|pair|real(pair[0])<=real(pair[1])));
                let mut adjoint=vectors.permute_axes(&[1,0]);adjoint.transform(|_,x|*x=conjugate(*x));
                let mut gram=a.clone();gram.gemm_2d::<Native>(&adjoint,&vectors,grid,algebra.one(),algebra.zero());
                let mut errors=[0.,0.];
                for(key,x)in gram.local_pairs(){
                    assert!(norm2(x).is_finite());
                    let identity=if key%n==key/n{algebra.one()}else{algebra.zero()};
                    errors[0]+=norm2(algebra.add(&x,&algebra.negate(&identity)));
                }
                vectors.transform(|key,x|*x=algebra.multiply(x,&eigenvalues[key/n]));
                let mut reconstructed=a.clone();
                reconstructed.gemm_2d::<Native>(&vectors,&adjoint,grid,algebra.one(),algebra.zero());
                for(key,x)in reconstructed.local_pairs(){
                    assert!(norm2(x).is_finite());errors[1]+=norm2(algebra.add(&x,&algebra.negate(&input(key))));
                }
                context.sum_f64(&mut errors);let bound=(n*n)as f64*1e-6;
                assert!(errors.iter().all(|e|e.sqrt()<=bound),"eigh n={n} degenerate={degenerate}: {errors:?} bound={bound}");
            }
        }
    }
}
scalar_case!(real32,f32,|r:f64,_:f64|r as f32,|x:f32|x,|x:f32|(x as f64).powi(2),|x:f32|x as f64);
scalar_case!(real64,f64,|r:f64,_:f64|r,|x:f64|x,|x:f64|x*x,|x:f64|x);
scalar_case!(complex32,Complex<f32>,|r:f64,i:f64|Complex::new(r as f32,i as f32),|x:Complex<f32>|x.conjugate(),|x:Complex<f32>|(x.re as f64).powi(2)+(x.im as f64).powi(2),|x:Complex<f32>|x.re as f64);
scalar_case!(complex64,Complex<f64>,|r:f64,i:f64|Complex::new(r,i),|x:Complex<f64>|x.conjugate(),|x:Complex<f64>|x.norm_squared(),|x:Complex<f64>|x.re);
fn run(c:&Context<'_>){real32(c);real64(c);complex32(c);complex64(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS typed_distributed_eigh: four scalar types, Hermitian and degenerate spectra, square subworld and original distribution, source Frobenius bounds; world+parity");}
    world.close();runtime.finalize();}
