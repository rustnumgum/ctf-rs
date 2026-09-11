use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

use ctf::{context::Context, cost::Models};

fn printed_precision_bound(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        0.5 * 10.0f64.powf(value.abs().log10().floor() - 4.0)
    }
}

fn check(context: &Context<'_>, output: PathBuf) {
    if context.rank() == 0 {
        fs::create_dir(&output).unwrap();
        fs::create_dir(output.join("dump")).unwrap();
    }
    context.barrier();

    let mut expected = Models::upstream(4);
    for model in expected.iter_mut() {
        let coefficients = model
            .coefficients()
            .iter()
            .map(|value| {
                if *value == 0.0 {
                    0.0
                } else {
                    value * 1.000012345
                }
            })
            .collect::<Vec<_>>();
        model.set_coefficients(&coefficients);
    }

    let coefficient_file = output.join("models.txt");
    if context.rank() == 0 {
        let mut file = File::create(&coefficient_file).unwrap();
        writeln!(file, "cuda_model 9.8765E+00").unwrap();
        drop(file);
        expected.write_all_models(&coefficient_file).unwrap();
    }
    context.barrier();

    let mut loaded = Models::upstream(4);
    loaded.load_all_models(&coefficient_file).unwrap();
    let mut maximum_ratio = 0.0f64;
    for model in expected.iter() {
        let actual = loaded.get(model.name()).coefficients();
        for (actual, expected) in actual.iter().zip(model.coefficients()) {
            let delta = (actual - expected).abs();
            let bound = printed_precision_bound(*expected);
            if *expected == 0.0 {
                assert_eq!(*actual, 0.0, "{} zero coefficient", model.name());
            } else {
                assert!(
                    delta <= bound,
                    "{} coefficient: expected={expected:E}, actual={actual:E}, delta={delta:E}, bound={bound:E}",
                    model.name()
                );
                maximum_ratio = maximum_ratio.max(delta / bound);
            }
        }
    }

    loaded
        .get_mut("csrred_mdl")
        .observe(context.rank() as f64 + 1.0, &[1.0, 2.0, 3.0]);
    loaded
        .dump_all_models(context, output.join("dump"))
        .unwrap();
    if context.rank() == 0 {
        let source = fs::read_to_string(&coefficient_file).unwrap();
        assert!(source.starts_with("cuda_model 9.8765E+00\n"));
        assert!(source.contains("csrred_mdl "));
        assert!(source.contains("E-"));

        let mut printed = Vec::new();
        loaded.print_all_models(&mut printed).unwrap();
        let printed = String::from_utf8(printed).unwrap();
        assert!(printed.starts_with("double csrred_mdl_init[] = {"));
        assert!(printed.contains("csrred_mdl is_tuned = 0 is_active = 1 (1)"));

        let dump = fs::read_to_string(output.join("dump").join("csrred_mdl")).unwrap();
        assert_eq!(dump.lines().count(), context.size() + 1);
        println!(
            "DIGIT / PASS model_io: Q=all registered coefficients; ref=in-memory values before write; bound=0.5*10^(floor(log10(abs(coef)))-4), zero exact; max d={maximum_ratio}; ranks={}",
            context.size()
        );
    }
}

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: model_io OUTPUT_DIRECTORY");
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    check(&world, output);
    world.close();
    drop(universe);
}
