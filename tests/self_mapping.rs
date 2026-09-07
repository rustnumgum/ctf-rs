use ctf::{mapping::Mapping,self_mapping::{map_self_indices,check_self_mapping}};
fn physical(axis:usize,np:usize)->Mapping{Mapping::Physical{axis,processes:np,child:Box::new(Mapping::Unmapped)}}
fn virtual_map(copies:usize)->Mapping{Mapping::Virtual{copies,child:Box::new(Mapping::Unmapped)}}
#[test]
fn source_first_pass_and_repeated_phase_coordination(){
    let mut maps=vec![Mapping::Unmapped,physical(0,3)];
    map_self_indices(&mut maps,&[7,7],&[false;4]).unwrap();
    assert_eq!(maps,vec![virtual_map(1),physical(0,3)]);
    assert!(!check_self_mapping(&maps,&[7,7]));
    map_self_indices(&mut maps,&[7,7],&[false;4]).unwrap();
    assert_eq!(maps,vec![virtual_map(3),physical(0,3)]);
    assert!(check_self_mapping(&maps,&[7,7]));
    let mut maps=vec![Mapping::Unmapped;3];
    map_self_indices(&mut maps,&[0,0,0],&[false;9]).unwrap();
    assert_eq!(maps,vec![virtual_map(1),virtual_map(1),Mapping::Unmapped]);
    assert!(check_self_mapping(&maps,&[0,0,0]));
}
#[test]
fn source_physical_descendants_and_repeated_children(){
    assert!(!check_self_mapping(&[physical(0,2),physical(0,2)],&[0,1]));
    assert!(!check_self_mapping(&[physical(0,2),virtual_map(2)],&[0,0]));
    let two=Mapping::Physical{axis:0,processes:2,child:Box::new(physical(1,2))};
    assert!(check_self_mapping(&[two.clone()],&[0]));
    assert!(!check_self_mapping(&[virtual_map(4),two],&[0,0]));
    let three=Mapping::Physical{axis:0,processes:2,child:Box::new(Mapping::Physical{
        axis:1,processes:2,child:Box::new(physical(2,2))})};
    assert!(!check_self_mapping(&[three],&[0])); // source compares every descendant to head+1
    assert!(check_self_mapping(&[],&[]));
    map_self_indices(&mut [],&[],&[]).unwrap();
}

#[test]
fn explicit_dimension_metadata_uses_source_floor_division(){
    let maps=[Mapping::Virtual{copies:3,child:Box::new(Mapping::Physical{axis:0,processes:2,
        child:Box::new(virtual_map(2))})},Mapping::Physical{axis:1,processes:5,child:Box::new(virtual_map(3))}];
    let dimensions=ctf::mapping::calc_dim(1000,&[25,19],&maps);
    assert_eq!(dimensions.virtual_size,55);
    assert_eq!(dimensions.virtual_edges,vec![2,1]);assert_eq!(dimensions.block_edges,vec![12,3]);
    assert_eq!(ctf::mapping::calc_dim(7,&[],&[]).virtual_size,7);
}
