use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::Distribution,tensor::Tensor};
fn value(key:usize,seed:usize)->f64{((key*seed%101)as f64-50.)/101.}
fn run(c:&Context<'_>){
    let d=Distribution::cyclic(vec![6,5,7,8],c.size());
    let mut a=Tensor::new(c,d.clone(),Arithmetic::<f64>::new());let mut b=a.clone();let mut output=a.clone();
    a.transform(|key,v|*v=value(key,17));b.transform(|key,v|*v=value(key,29));output.transform(|key,v|*v=value(key,37));
    output.transform_from("ijkl",&a,"ijkl",&b,"ijkl",|a,b,c|*c=a**c*a+b**c*b);
    for(key,actual)in output.local_pairs(){let(a,b,old)=(value(key,17),value(key,29),value(key,37));
        assert!(actual.is_finite()&&(actual-(a*old*a+b*old*b)).abs()<1e-6);
    }
    assert_eq!(output.distribution(),&d);
    // Additional direct-Rust coverage, separate from the upstream identity:
    // typed input, scalar broadcast, repeated output and empty local shards.
    let mut integers=Tensor::new(c,Distribution::cyclic(vec![2],c.size()),Arithmetic::<i64>::new());
    integers.transform(|key,v|*v=key as i64+1);
    let mut scalar=Tensor::new(c,Distribution::cyclic(vec![],c.size()),Arithmetic::<f64>::new());
    scalar.transform(|_,v|*v=0.5);
    let mut diagonal=Tensor::new(c,Distribution::cyclic(vec![2,3,3],c.size()),Arithmetic::<f64>::new());
    diagonal.transform(|_,v|*v=5.);
    diagonal.transform_from("ijj",&integers,"i",&scalar,"",|a,b,c|*c+=*a as f64*b);
    for(key,actual)in diagonal.local_pairs(){
        let expected=if key/2%3==key/6{5.+(key%2+1)as f64*0.5}else{5.};
        assert_eq!(actual,expected);
    }
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS upstream_bivar_transform: original mutable-output identity, deterministic global-key values, abs(error)<1e-6; world+parity");}
    world.close();runtime.finalize();}
