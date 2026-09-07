use ctf::{sparse_matricize::{Matricization,Dematricization,matricize_pairs,dematricize_pairs},symmetry::Symmetry};

#[test]
fn phase_and_axis_order_with_sorted_reverse() {
    let layout=Matricization { shape:vec![5,3,2],padded_shape:vec![6,3,2],
        links:vec![Symmetry::NS;3],folded_shape:vec![3,3,2],reverse_ordering:vec![2,0,1],
        row_dimensions:2,phases:vec![2,1,1] };
    let coo=matricize_pairs(&layout,&[(28,7_i64),(1,0),(16,-2)]);
    assert_eq!(coo.shape(),(6,3));
    assert_eq!(coo.entries(),&[(4,3,7),(1,1,0),(2,1,-2)]);
    let reverse=Dematricization { shape:layout.shape,reverse_ordering:layout.reverse_ordering,
        row_dimensions:2,phases:layout.phases,phase_ranks:vec![1,0,0] };
    assert_eq!(dematricize_pairs(&reverse,&coo),vec![(1,0),(16,-2),(28,7)]);
}
#[test]
fn folded_symmetric_binomial_coordinates() {
    let layout=Matricization { shape:vec![3,3,2],padded_shape:vec![3,3,2],
        links:vec![Symmetry::SY,Symmetry::NS,Symmetry::NS],folded_shape:vec![6,2],
        reverse_ordering:vec![1,0],row_dimensions:1,phases:vec![1;3] };
    let coo=matricize_pairs(&layout,&[(0,0_i64),(3,4),(17,9)]);
    assert_eq!(coo.shape(),(2,6));assert_eq!(coo.entries(),&[(1,1,0),(1,2,4),(2,6,9)]);
    let mut phased=layout; phased.shape=vec![5,5,2];phased.padded_shape=vec![6,6,2];phased.phases=vec![2,2,1];
    let coo=matricize_pairs(&phased,&[(41,5_i64),(18,6)]);
    assert_eq!(coo.entries(),&[(2,2,5),(1,3,6)]);
}
#[test]
fn scalar_and_empty_matricization() {
    let layout=Matricization { shape:vec![],padded_shape:vec![],links:vec![],folded_shape:vec![],
        reverse_ordering:vec![],row_dimensions:0,phases:vec![] };
    let coo=matricize_pairs(&layout,&[(0,0_i64)]);assert_eq!(coo.shape(),(1,1));
    assert_eq!(coo.entries(),&[(1,1,0)]);
    let reverse=Dematricization { shape:vec![],reverse_ordering:vec![],row_dimensions:0,phases:vec![],phase_ranks:vec![] };
    assert_eq!(dematricize_pairs(&reverse,&coo),vec![(0,0)]);
    let coo=matricize_pairs(&layout,&[] as &[(usize,i64)]);
    assert!(coo.entries().is_empty());assert!(dematricize_pairs(&reverse,&coo).is_empty());
}
