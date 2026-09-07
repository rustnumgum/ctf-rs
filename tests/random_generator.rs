use ctf::random::Generator;
#[test]
fn pinned_mt19937_64_streams_cross_twist_boundaries(){
    // Development-only std::mt19937_64 oracle, matching common.cxx's parameters.
    for(seed,expected)in[
        (0,[2947667278772165694,18301848765998365067,11228354904504431959,17661967264253682746,12220678344985132467,13999015252384676179]),
        (1,[2469588189546311528,2516265689700432462,7051797671038026992,4522861927766102283,18157062757242514893,37632317631463696]),
        (5489,[14514284786278117030,4620546740167642908,1370093900783164344,6776537281339823025,15547153445796060183,12329720415526259303])]{
        let mut rng=Generator::new(seed);let mut found=Vec::new();
        for index in 0..=624{let value=rng.next_u64();if [0,1,311,312,623,624].contains(&index){found.push(value);}}
        assert_eq!(found,expected);
    }
    let mut rng=Generator::new(0);
    assert_eq!(rng.unit_interval().to_bits(),(2947667278772165694u64 as f64/u64::MAX as f64).to_bits());
}
