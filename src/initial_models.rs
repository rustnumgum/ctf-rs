// CPU coefficients copied from pinned cc4s CTF shared/init_models.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Upstream static seeds, not calibration measurements for the current machine.
pub const CPU: &[(&str, &[f64])] = &[
    ("csrred_mdl", &[3.0689E-03, 2.2385E-03, 4.4815E-07]),
    ("csrred_mdl_cst", &[-1.8323E-04, 1.3076E-04, 2.8732E-09]),
    ("alltoall_mdl", &[1.0000E-06, 1.0000E-06, 5.0000E-10]),
    ("alltoallv_mdl", &[7.3164E-23, 1.0404E-04, 2.5827E-07]),
    ("red_mdl", &[4.5530E-11, 3.0466E-17, 2.5E-9]),
    ("red_mdl_cst", &[1.2881E-04, 1.4093E-16, 8.3976E-10]),
    ("allred_mdl", &[4.7939E-14, 7.4715E-13, 2.0949E-06]),
    ("allred_mdl_cst", &[-3.3754E-04, 2.1343E-04, 3.0801E-09]),
    ("bcast_mdl", &[1.1115E-16, 1.0754E-16, 1.32E-9]),
    ("seq_tsr_ctr_mdl_cst", &[7.8076E-13, 6.9558E-08, 1.3923E-08]),
    ("seq_tsr_ctr_mdl_ref", &[4.9138E-08, 5.8290E-10, 4.8575E-11]),
    ("seq_tsr_ctr_mdl_inr", &[6.0166E-21, 2.3443E-13, 1.4286E-11]),
    ("seq_tsr_ctr_mdl_cst_inr", &[0.0, 0.0, 1.6E-11]),
    ("long_contig_transp_mdl", &[0.0, 1.0E-08]),
    ("shrt_contig_transp_mdl", &[0.0, 1.5E-08]),
    ("non_contig_transp_mdl", &[2.6680E-05, 4.6247E-08]),
    (
        "seq_tsr_spctr_cst_k0",
        &[5.3745E-06, 3.6464E-08, 2.2334E-10],
    ),
    (
        "seq_tsr_spctr_cst_k1",
        &[5.3745E-06, 3.6464E-08, 2.2334E-10],
    ),
    (
        "seq_tsr_spctr_cst_k2",
        &[2.1303E-74, 5.7379E-09, 4.1887E-11],
    ),
    (
        "seq_tsr_spctr_cst_k3",
        &[1.4917E-05, 2.5510E-10, 5.4110E-12],
    ),
    (
        "seq_tsr_spctr_cst_k4",
        &[5.6408E-06, 1.8318E-09, 5.2399E-80],
    ),
    (
        "seq_tsr_spctr_cst_k5",
        &[2.8218E-05, 3.0049E-09, 5.2399E-11],
    ),
    ("seq_tsr_spctr_k0", &[3.9315E-05, 2.2285E-08, 6.1958E-08]),
    ("seq_tsr_spctr_k1", &[5.3745E-06, 3.6464E-08, 2.2334E-10]),
    ("seq_tsr_spctr_k2", &[5.9868E-14, 1.4877E-09, 5.3514E-12]),
    ("seq_tsr_spctr_k3", &[1.3994E-15, 2.5071E-09, 2.7323E-11]),
    ("seq_tsr_spctr_k4", &[2.0404E-04, 8.2989E-09, 6.0431E-11]),
    ("seq_tsr_spctr_k5", &[6.9073E-15, 4.0130E-09, 2.2669E-13]),
    ("pin_keys_mdl", &[4.0261E-05, 7.2443E-07]),
    ("spredist_mdl", &[6.8713E-23, 7.8867E-04, 6.9422E-11]),
    ("dgtog_res_mdl", &[0.0, 0.0, 7.25E-10]),
    ("blres_mdl", &[0.0, 1E-10]),
];
