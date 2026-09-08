use ctf::{
    context::Context,
    linalg::Native,
    model::{CubicModel, LinearModel, cubic_features},
};
fn train(context: &Context<'_>) {
    let mut ring = LinearModel::new("ring", vec![2., 3.], 2);
    assert_eq!(ring.estimate(&[1., 2.]), 8.);
    ring.observe(8., &[1., 2.]);
    ring.observe(11., &[1., 3.]);
    ring.observe(14., &[1., 4.]);
    assert_eq!(
        ring.retained_observations(),
        &[(14., vec![1., 4.]), (11., vec![1., 3.])]
    );
    assert!(!ring.update::<Native>(context).unwrap());
    assert_eq!(ring.diagnostics().average_total_time, 33.);
    assert_eq!(ring.diagnostics().average_over_time, 0.);
    assert!(ring.should_observe());
    let mut clipped = LinearModel::new("clip", vec![-2.], 1);
    assert_eq!(clipped.estimate(&[3.]), 0.);
    clipped.observe(4., &[3.]);
    assert!(!clipped.update::<Native>(context).unwrap());
    assert_eq!(clipped.diagnostics().average_over_time, 4.);

    for (uneven, rank_deficient) in [(false, false), (true, false), (false, true)] {
        let count = if uneven {
            if context.rank() == 0 {
                64 * context.size()
            } else {
                0
            }
        } else {
            64
        };
        let mut model = LinearModel::new("fit", vec![1., 1.], 64 * context.size());
        for i in 0..count {
            let x = (i + 1) as f64;
            let p = if rank_deficient { [x, x] } else { [1., x] };
            model.observe(2. * p[0] + 3. * p[1], &p);
        }
        assert!(model.update::<Native>(context).unwrap());
        assert!(model.diagnostics().tuned);
        let mut residual = 0.;
        for i in 0..count {
            let x = (i + 1) as f64;
            let p = if rank_deficient { [x, x] } else { [1., x] };
            residual += (model.estimate(&p) - (2. * p[0] + 3. * p[1])).powi(2);
        }
        let mut all = [residual];
        context.sum_f64(&mut all);
        // Same Frobenius reconstruction bound as upstream QR: m*n*n*1e-6.
        let bound = (64 * context.size()) as f64 * 4e-6;
        assert!(
            all[0].sqrt() <= bound,
            "model fit residual {} exceeds {bound}",
            all[0].sqrt()
        );
        if context.rank() == 0 {
            println!(
                "model training uneven={uneven}, rank_deficient={rank_deficient}: residual={}, bound={bound}",
                all[0].sqrt()
            );
        }
    }
    assert_eq!(
        cubic_features(&[2., 3.]),
        vec![2., 3., 4., 6., 9., 8., 12., 18., 27.]
    );
    let mut cubic = CubicModel::new("cubic", 2, vec![1.; 9], 2);
    assert_eq!(cubic.estimate(&[2., 3.]), 89.);
    cubic.observe(89., &[2., 3.]);
    assert!(!cubic.update::<Native>(context).unwrap());
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    train(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    train(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS model_training: history/features exact; QR/SVD fits within upstream reconstruction bound; ranks={}",
            world.size()
        );
    }
    world.close();
    drop(universe);
}
