use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Topology},tensor::Tensor};
fn run(c:&Context<'_>){
    let mut a=Tensor::new(c,Distribution::cyclic(vec![6,5,7,8],c.size()),Arithmetic::<f64>::new());
    let mut b=a.clone();
    a.transform(|key,v|*v=((key*17%101) as f64-50.)/101.);
    b.transform(|key,v|*v=((key*29%103) as f64-51.)/103.);
    let old=a.clone();
    a.contract_function_on("ijkl",&old,"ijkl",&b,"ijkl",Topology::new(vec![c.size()]),"i",
        1.,0.5,|a,b|a*b+b*a);
    for(key,actual)in a.local_pairs(){
        let x=((key*17%101) as f64-50.)/101.;let y=((key*29%103) as f64-51.)/103.;
        let expected=0.5*x+x*y+y*x;
        assert!(actual.is_finite()&&(actual-expected).abs()<1e-6);
    }
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS upstream_bivar_function: source four-dimensional identity, deterministic global-key fixture; abs(error)<1e-6; world+parity");}
    world.close();runtime.finalize();}
