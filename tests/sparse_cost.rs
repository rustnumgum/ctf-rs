use ctf::{cost::Models, sparse_cost::local::{Folded, Kernel, Local}};
use ctf::sparse_cost::{Fractions,Storage,Operand2d,TwoDimensional,ReplicaOperand,Replication,Virtual};

fn models() -> Models {
    let mut m=Models::upstream(1);
    for (name,coeff) in [("bcast_mdl",[1.,2.,3.]),("csrred_mdl",[7.,5.,2.]),
        ("csrred_mdl_cst",[999.;3]),("red_mdl",[11.,7.,3.]),("red_mdl_cst",[19.,11.,5.])] {
        m.get_mut(name).set_coefficients(&coeff);
    }
    m
}
fn storage(sparse:bool,pair_size:usize,dense_virtual_size:usize)->Storage {
    Storage { sparse,element_size:2,pair_size,dense_virtual_size,custom_addition:false }
}
#[test]
fn two_dimensional_payload_memory_and_layers() {
    let m=models(); let f=Fractions { a:0.125,b:0.25,c:0.0625 };
    let mut level=TwoDimensional { edge:4,
        a:Operand2d { storage:storage(true,3,5),moving:true,ranks:4,outer:2,inner:3 },
        b:Operand2d { storage:storage(false,3,5),moving:false,ranks:2,outer:2,inner:2 },
        c:Operand2d { storage:storage(true,5,7),moving:true,ranks:2,outer:2,inner:1 } };
    assert_eq!(level.footprint(f),32); // trunc(11.25) + 8 + trunc(13.125)
    assert_eq!(level.temp(f),4); assert_eq!(level.memory(f,10),42);
    assert_eq!(level.memory(f,0),36);
    for (layers,multiplier) in [(1,4.0),(2,2.0),(8,1.0)] {
        assert_eq!(level.fixed_time(&m,f,layers),58.0*multiplier);
        assert_eq!(level.est_time(&m,f,layers,3.0),61.0*multiplier);
    }
    level.c.storage.custom_addition=true;
    assert_eq!(level.fixed_time(&m,f,2),116.0); // sparse always csrred_mdl
    level.c.storage.sparse=false;
    assert_eq!(level.footprint(f),23); assert_eq!(level.temp(f),0);
    assert_eq!(level.est_time(&m,f,2,3.0),142.0); // 2*(38 + 30 + 3)
    level.c.storage.custom_addition=false;
    assert_eq!(level.est_time(&m,f,2,3.0),118.0);
    // No sparse panel storage for stationary whole or contiguous views.
    level.a.moving=false; level.a.outer=1;
    assert_eq!(level.footprint(f),12);
    level.a.outer=2; level.a.inner=0;
    assert_eq!(level.footprint(f),12);
    level.a.inner=3; level.c.moving=false;
    assert_eq!(level.fixed_time(&m,f,1),0.0);
    assert_eq!(level.footprint(f),23);
}
#[test]
fn replication_and_virtual_recursive_composition() {
    let m=models(); let f=Fractions { a:0.125,b:0.25,c:0.0625 };
    let mut level=Replication {
        a:ReplicaOperand { storage:storage(true,3,99),size:5,communicator_ranks:vec![2,4] },
        b:ReplicaOperand { storage:storage(true,3,99),size:7,communicator_ranks:vec![2] },
        c:ReplicaOperand { storage:storage(true,5,99),size:9,communicator_ranks:vec![4] } };
    assert_eq!(level.fixed_time(&m,f),53.0); assert_eq!(level.est_time(&m,f,3.0),56.0);
    assert_eq!(level.footprint(f),9); assert_eq!(level.temp(),0);
    assert_eq!(level.memory(f,10),19);
    level.a.communicator_ranks.pop(); level.b.communicator_ranks.push(4);
    assert_eq!(level.footprint(f),13); // B now owns copy; A no longer does
    level.c.storage.sparse=false;
    assert_eq!(level.temp(),18); assert_eq!(level.footprint(f),5);
    assert_eq!(level.memory(f,10),23);
    level.c.communicator_ranks.clear();
    assert_eq!(level.temp(),0); assert_eq!(level.memory(f,10),15);
    let virtual_level=Virtual { dimensions:vec![2,3,4],orders:[3,2,3] };
    assert_eq!(virtual_level.footprint(),80);
    assert_eq!(virtual_level.est_time(level.est_time(&m,f,3.0)),24.0*level.est_time(&m,f,3.0));
    assert_eq!(virtual_level.memory(level.memory(f,10)),95);
}

#[test]
fn cpu_leaf_models_and_source_traffic() {
    let mut models = Models::upstream(1);
    for custom in [false, true] { for id in 0..6 {
        let name = if custom { format!("seq_tsr_spctr_cst_k{id}") }
            else { format!("seq_tsr_spctr_k{id}") };
        models.get_mut(&name).set_coefficients(&[id as f64+1.0+if custom { 100.0 } else { 0.0 },2.0,4.0]);
    }}
    let kernels = [Kernel::General { extents:vec![2,3,4] },
        Kernel::Folded { kind:Folded::CooDense,m:2,n:3,k:4 },
        Kernel::Folded { kind:Folded::CsrDense,m:2,n:3,k:4 },
        Kernel::Folded { kind:Folded::CsrSparseDense,m:2,n:3,k:4 },
        Kernel::Folded { kind:Folded::CsrSparse,m:2,n:3,k:4 },
        Kernel::Folded { kind:Folded::CcsrDense,m:2,n:3,k:4 }];
    for custom in [false,true] { for (id,kernel) in kernels.iter().enumerate() {
        let leaf=Local { kernel:kernel.clone(),custom,packed_elements:[3,5,7],
            element_bytes:[1;3],sparse:[true;3],nnz_fraction:[0.125,0.0625,0.03125] };
        let e=leaf.estimate(&models);
        assert_eq!(e.flops,3.375);
        assert_eq!(e.memory_traffic_bytes,if id==0 { 12 } else { 106 });
        assert_eq!(e.seconds,id as f64+1.0+if custom { 100.0 } else { 0.0 }
            +2.0*e.memory_traffic_bytes as f64+13.5);
    }}
    let mut leaf=Local { kernel:Kernel::General { extents:vec![2,0,4] },custom:false,
        packed_elements:[0,0,8],element_bytes:[8;3],sparse:[true,false,false],nnz_fraction:[0.0;3] };
    let e=leaf.estimate(&models);
    assert_eq!(e.flops,0.0); assert_eq!(e.memory_traffic_bytes,64);
    leaf.kernel=Kernel::General { extents:vec![] };
    leaf.sparse=[false;3]; leaf.packed_elements=[1;3];
    let e=leaf.estimate(&models);
    assert_eq!(e.flops,2.0); assert_eq!(e.memory_traffic_bytes,24);
}
