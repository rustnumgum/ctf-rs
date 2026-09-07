// Existing upstream test_la.py L1/triangle acceptance, applied to typed factors.
use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring,Wire},context::{Context,Runtime},
    mapping::{Distribution,Topology},tensor::Tensor};

fn product<'c,'r,T:Clone+PartialEq+Wire>(a:&Tensor<'c,'r,Arithmetic<T>>,b:&Tensor<'c,'r,Arithmetic<T>>,
    c:&mut Tensor<'c,'r,Arithmetic<T>>,grid:[usize;2])
where Arithmetic<T>:Semiring<Element=T>+Clone{
    let algebra=Arithmetic::<T>::new();
    c.contract_from_on_grid("ij",a,"ik",b,"kj",Topology::new(grid.to_vec()),algebra.one(),algebra.zero()).unwrap();
}
fn residual<T:Clone+PartialEq+Wire>(expected:&Tensor<'_,'_,Arithmetic<T>>,actual:&Tensor<'_,'_,Arithmetic<T>>,
    norm:impl Fn(T)->f64,use_actual_norm:bool)
where Arithmetic<T>:ctf::algebra::Group<Element=T>{
    let algebra=Arithmetic::<T>::new();let mut norms=[0.,0.];
    for((key,a),(other,b))in expected.local_pairs().into_iter().zip(actual.local_pairs()){
        assert_eq!(key,other);assert!(norm(b.clone()).is_finite());
        norms[0]+=norm(algebra.add(&a,&algebra.negate(&b)));
        norms[1]+=norm(if use_actual_norm{b}else{a});
    }
    expected.context().sum_f64(&mut norms);
    assert!(norms[0]<=1e-3||norms[0]/norms[1]<=1e-3,"L1 {} / {}",norms[0],norms[1]);
}
macro_rules! scalar_case{
    ($name:ident,$t:ty,$value:expr,$conjugate:expr,$norm:expr)=>{
        fn $name(context:&Context<'_>){
            let value=$value;let conjugate=$conjugate;let norm=$norm;
            let algebra=Arithmetic::<$t>::new();let np=context.size();
            let grid=if np==4{[2,2]}else{[np,1]};
            let make=|m,n|Tensor::new(context,Distribution::cyclic(vec![m,n],np),algebra.clone());
            for n in [1,5]{
                let mut a=make(n,n);
                a.transform(|key,out|{
                    let i=key%n;let j=key/n;let mut sum=algebra.zero();
                    for k in 0..=i.min(j){
                        let l=if i==k{value((i+2)as f64,0.)}else{value(1.,1.)};
                        let r=if j==k{value((j+2)as f64,0.)}else{value(1.,1.)};
                        sum=algebra.add(&sum,&algebra.multiply(&l,&conjugate(r)));
                    }*out=sum;
                });
                for lower in [true,false]{
                    let factor=a.cholesky(grid,lower).unwrap();assert_eq!(factor.distribution(),a.distribution());
                    let mut adjoint=factor.permute_axes(&[1,0]);adjoint.transform(|_,x|*x=conjugate(*x));
                    let mut reconstructed=make(n,n);
                    if lower{product(&factor,&adjoint,&mut reconstructed,grid)}else{product(&adjoint,&factor,&mut reconstructed,grid)}
                    residual(&a,&reconstructed,norm,false);
                    let mut forbidden=[0.];for(key,x)in factor.local_pairs(){
                        if(lower&&key%n<key/n)||(!lower&&key%n>key/n){forbidden[0]+=norm(x).powi(2);}
                    }context.sum_f64(&mut forbidden);assert!(forbidden[0].sqrt()<=1e-6);
                    for left in [true,false]{for transpose in [false,true]{
                        let(m,cols)=if left{(n,3)}else{(3,n)};let mut rhs=make(m,cols);
                        rhs.transform(|key,x|*x=value((key+1)as f64/8.,-0.5));
                        let solution=rhs.solve_tri(&factor,grid,lower,left,transpose).unwrap();
                        assert_eq!(solution.distribution(),rhs.distribution());
                        let op=if transpose{factor.permute_axes(&[1,0])}else{factor.clone()};
                        let mut reconstructed=make(m,cols);
                        if left{product(&op,&solution,&mut reconstructed,grid)}else{product(&solution,&op,&mut reconstructed,grid)}
                        residual(&rhs,&reconstructed,norm,false);
                    }}
                }
                for nrhs in [1,9]{
                    let mut rhs=make(n,nrhs);rhs.transform(|key,x|*x=value((key%13+1)as f64/7.,0.5));
                    let solution=rhs.solve_spd(&a).unwrap();assert_eq!(solution.distribution(),rhs.distribution());
                    let mut reconstructed=make(n,nrhs);product(&a,&solution,&mut reconstructed,grid);
                    residual(&rhs,&reconstructed,norm,true);
                }
            }
        }
    }
}
scalar_case!(real32,f32,|r:f64,_:f64|r as f32,|x:f32|x,|x:f32|x.abs()as f64);
scalar_case!(real64,f64,|r:f64,_:f64|r,|x:f64|x,|x:f64|x.abs());
scalar_case!(complex32,Complex<f32>,|r:f64,i:f64|Complex::new(r as f32,i as f32),|x:Complex<f32>|x.conjugate(),|x:Complex<f32>|x.norm_squared().sqrt()as f64);
scalar_case!(complex64,Complex<f64>,|r:f64,i:f64|Complex::new(r,i),|x:Complex<f64>|x.conjugate(),|x:Complex<f64>|x.norm_squared().sqrt());
fn run(c:&Context<'_>){real32(c);real64(c);complex32(c);complex64(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS typed_matrix_factors: native four-type distributed POTRF/POSV/TRSM, Hermitian reconstruction, padding and zero local rows; world+parity; upstream L1<=1e-3 or relative<=1e-3, triangle<=1e-6");}
    world.close();runtime.finalize();}
