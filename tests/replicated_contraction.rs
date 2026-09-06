use ctf::{context::Runtime,contraction::{replicated_f64,virtualized},algebra::Arithmetic};
fn main() {
    let mut c = [10i64];
    virtualized(&Arithmetic::<i64>::new(),&[1],&[3],"k",&[1,2,3],&[1],&[3],"k",&[4,5,6],
        &[],&[],"",&mut c,&2,&3);
    assert_eq!(c,[94]);
    let runtime = Runtime::initialize(); let world = runtime.world();
    let mut a = [world.rank() as f64+1.]; let mut b = [2.]; let mut c = [10.];
    replicated_f64(&[],&[],&[&world],&[],&[],"",&mut a,&[],&[],"",&mut b,&[],&[],"",&mut c,2.,3.);
    if world.rank() == 0 {assert_eq!(c,[30.+2.*(world.size()*(world.size()+1)) as f64]);}
    let mut a = if world.rank() == 0 {[2.,3.]} else {[99.,99.]};
    let mut b = if world.rank() == 0 {[4.,5.]} else {[88.,88.]}; let mut c = [0.];
    replicated_f64(&[&world],&[&world],&[],&[2],&[1],"k",&mut a,&[2],&[1],"k",&mut b,&[],&[],"",&mut c,1.,0.);
    assert_eq!(c,[23.]);
    if world.rank() != 0 {assert_eq!(a,[0.,0.]);assert_eq!(b,[0.,0.]);}
    if world.rank() == 0 {println!("DIGIT / PASS replicated_contraction: virtual beta, Reduce roots, broadcast cleanup; ranks={}",world.size());}
    world.close();runtime.finalize();
}
