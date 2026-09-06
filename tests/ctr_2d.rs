use ctf::{context::Runtime,ctr_2d::{execute,Layers,Panel}};
fn main() {
    let runtime = Runtime::initialize(); let world = runtime.world();
    let np = world.size(); let rank = world.rank(); let edge = 2*np;
    let whole = Panel {comm:None,outer:1,inner:0};
    let moving = Panel {comm:Some(&world),outer:1,inner:2};
    let a: Vec<_> = [rank,rank+np].into_iter().flat_map(|step|[(step+1) as f64,(step+2) as f64]).collect();
    let mut c = [10.];
    execute(edge,Layers {count:1,index:0},moving,whole,whole,&a,&[1.,2.],&mut c,3.,|a,b,c,beta,layer| {
        assert_eq!(layer,Layers {count:1,index:0}); c[0] = beta*c[0]+a[0]*b[0]+a[1]*b[1];
    });
    assert_eq!(c,[30.+(3*edge*(edge-1)/2+5*edge) as f64]);
    // Moving output with two strips and cyclic root ownership; beta on scatter.
    let output = Panel {comm:Some(&world),outer:2,inner:1}; let mut c = [10.;4];
    let input = Panel {comm:None,outer:1,inner:1}; let a: Vec<_> = (1..=edge).map(|i|i as f64).collect();
    execute(edge,Layers {count:1,index:0},input,whole,output,&a,&[1.],&mut c,2.,|a,_,c,beta,_| {
        c[0] = beta*c[0]+a[0]; c[1] = beta*c[1]+2.*a[0];
    });
    assert_eq!(c,[20.+np as f64*(rank+1) as f64,20.+np as f64*(rank+np+1) as f64,
        20.+2.*np as f64*(rank+1) as f64,20.+2.*np as f64*(rank+np+1) as f64]);
    if rank == 0 {println!("DIGIT / PASS ctr_2d: cyclic panel broadcasts, moving-output Reduce, strided scatter; ranks={np}");}
    world.close();runtime.finalize();
}
