use ctf::algebra::{Arithmetic,Group,Monoid,Semiring,Wire};
#[test]
fn source_promoted_narrow_integer_arithmetic_and_wire(){
    let a=Arithmetic::<i8>::new();assert_eq!(a.add(&127,&1),-128);
    assert_eq!(a.multiply(&100,&3),44);assert_eq!(a.negate(&-128),-128);
    let a=Arithmetic::<i16>::new();assert_eq!(a.add(&32767,&1),-32768);
    assert_eq!(a.multiply(&30000,&3),24464);assert_eq!(a.negate(&-32768),-32768);
    let mut wire=Vec::new();(-17i8).encode(&mut wire);(-12345i16).encode(&mut wire);
    assert_eq!(wire,vec![239,199,207]);assert_eq!(i8::decode(&wire[..1]),-17);
    assert_eq!(i16::decode(&wire[1..]),-12345);
    let a=Arithmetic::<bool>::new();for x in [false,true]{for y in [false,true]{
        assert_eq!(a.add(&x,&y),x||y);assert_eq!(a.multiply(&x,&y),x&&y);
    }}
}
