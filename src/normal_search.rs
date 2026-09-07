// Adapted from cc4s contraction.cxx normal mapping search, lines 2834-2913.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{context::Context,mapping::{Distribution,Mapping,Topology},normal_mapping::{Problem,ProblemError}};

#[derive(Clone,Debug)]
pub struct Candidate {
    pub source_id: usize,
    pub permutation: usize,
    pub template: usize,
    pub distributions: [Distribution;3],
}

fn candidate(problem:&Problem,indices:[&str;3],old:[Option<&Distribution>;3],
    catalog:&[Topology],template:usize,permutation:usize)->Option<Candidate>{
    let mut initial:[Option<&[Mapping]>;3]=[None;3];
    let topology=if template<8{
        let mut selected:Option<&Topology>=None;
        for operand in 0..3{if template&(1<<operand)!=0{
            let distribution=old[operand]?;
            if selected.is_some_and(|topology|topology!=&distribution.topology){return None;}
            selected=Some(&distribution.topology);initial[operand]=Some(&distribution.mappings);
        }}
        selected?
    }else{&catalog[template-8]};
    let distributions=problem.map_to_topology(topology,permutation,initial).ok()?;
    if !crate::mapping_preflight::check(distributions.each_ref(),indices){return None;}
    Some(Candidate{source_id:6*template+permutation,permutation,template,distributions})
}

/// Rebuild an already selected normal ID on every rank, without enumerating
/// other candidates or broadcasting tensor data. Invalid selected IDs are errors,
/// as in the source final-map assertion (contraction.cxx:3364-3396).
pub fn reconstruct(source_id:usize,shapes:[&[usize];3],indices:[&str;3],
    old:[Option<&Distribution>;3],catalog:&[Topology])->Result<Candidate,ProblemError>{
    let problem=Problem::new(shapes,indices)?;
    let template=source_id/6;
    assert!((1..catalog.len()+8).contains(&template),"invalid selected normal mapping ID");
    Ok(candidate(&problem,indices,old,catalog,template,source_id%6)
        .expect("selected normal mapping failed final preflight"))
}

/// Visit this rank's normal candidates in source permutation/template order.
/// Templates 1..7 retain old-layout bit subsets; 8 onward use catalog layouts.
/// Missing or inconsistent old topology subsets and source mapping negatives
/// are ordinary candidate rejections. No costs or memory limits are fabricated.
pub fn visit_local(context:&Context<'_>,shapes:[&[usize];3],indices:[&str;3],
    old:[Option<&Distribution>;3],catalog:&[Topology],mut visit:impl FnMut(Candidate))
    ->Result<(),ProblemError>{
    let problem=Problem::new(shapes,indices)?;
    for operand in 0..3{if let Some(d)=old[operand]{assert_eq!(d.shape,shapes[operand]);assert_eq!(d.topology.size(),context.size());}}
    assert!(catalog.iter().all(|topology|topology.size()==context.size()));
    for permutation in 0..6{
        for template in (context.rank()+1..catalog.len()+8).step_by(context.size()){
            if let Some(candidate)=candidate(&problem,indices,old,catalog,template,permutation){visit(candidate);}
        }
    }
    Ok(())
}
