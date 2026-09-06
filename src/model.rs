// Adapted from cc4s CTF shared/model.cxx. Copyright (c) 2011,
// Edgar Solomonik. See LICENSE.
//! Performance model prediction, rolling observations and distributed QR/SVD
//! training. Updates communicate only the reduced systems, not all observations.
use crate::{context::Context,linalg::LocalKernels};

#[derive(Clone,Debug,Default)]
pub struct Diagnostics {
    pub observations:usize,
    pub tuned:bool,
    pub average_total_time:f64,
    /// Source calls prediction below actual "over_time".
    pub average_over_time:f64,
    /// Source calls prediction above actual "under_time".
    pub average_under_time:f64,
}
pub struct LinearModel {
    name:String,
    coefficients:Vec<f64>,
    history:Vec<(f64,Vec<f64>)>,
    capacity:usize,
    diagnostics:Diagnostics,
    totals:[f64;3],
}
impl LinearModel {
    pub fn new(name:impl Into<String>,coefficients:Vec<f64>,history_size:usize)->Self {
        assert!(!coefficients.is_empty() && history_size>0);
        Self {name:name.into(),coefficients,history:Vec::new(),capacity:history_size,
            diagnostics:Diagnostics::default(),totals:[0.;3]}
    }
    pub fn name(&self)->&str {&self.name}
    pub fn coefficients(&self)->&[f64] {&self.coefficients}
    pub fn set_coefficients(&mut self,values:&[f64]) {
        assert_eq!(values.len(),self.coefficients.len());self.coefficients.copy_from_slice(values);
    }
    pub fn diagnostics(&self)->&Diagnostics {&self.diagnostics}
    pub fn retained_observations(&self)->&[(f64,Vec<f64>)] {&self.history}
    /// Pinned source's deactivate predicate contains threshold<threshold, hence
    /// training never automatically deactivates. Do not silently change that rule.
    pub fn should_observe(&self)->bool {true}
    pub fn estimate(&self,parameters:&[f64])->f64 {
        assert_eq!(parameters.len(),self.coefficients.len());
        parameters.iter().zip(&self.coefficients).map(|(x,c)|x*c).sum::<f64>().max(0.)
    }
    pub fn observe(&mut self,seconds:f64,parameters:&[f64]) {
        assert!(seconds>=0.);let predicted=self.estimate(parameters);
        self.totals[0]+=seconds;
        if predicted>seconds {self.totals[2]+=predicted-seconds;} else {self.totals[1]+=seconds-predicted;}
        let slot=self.diagnostics.observations%self.capacity;
        let observation=(seconds,parameters.to_vec());
        if self.history.len()<self.capacity {self.history.push(observation);} else {self.history[slot]=observation;}
        self.diagnostics.observations+=1;
    }
    /// Collective model update, source threshold 16*np*nparam. Ranks with fewer
    /// than nparam retained observations contribute zero reduced systems.
    /// Native failures surface as LAPACK info; no alternate solver is attempted.
    pub fn update<K:LocalKernels>(&mut self,context:&Context<'_>)->Result<bool,i32> {
        let n=self.coefficients.len();let np=context.size();let count=self.history.len();
        let count=context.all_reduce(&crate::algebra::Arithmetic::<i64>::new(),&(count as i64)) as usize;
        let tuned=count>=16*np*n;
        if tuned {
            let (r,y)=if self.history.len()<n {(vec![0.;n*n],vec![0.;n])} else {
                let m=self.history.len()+n;let mut a=vec![0.;m*n];let mut b=vec![0.;m];
                for j in 0..n {
                    a[j+j*m]=if self.coefficients[j]!=0. {
                        1e6f64.min(self.diagnostics.average_total_time/self.coefficients[j]/1000.)
                    } else {1.};
                }
                for (i,(seconds,parameters)) in self.history.iter().enumerate() {
                    b[n+i]=*seconds;for j in 0..n {a[n+i+j*m]=parameters[j];}
                }
                K::qr_reduce(m,n,&a,&b)?
            };
            // Source sub_np=np: explicit duplicate communicator and allgathers.
            let comm=context.split(Some(1),context.rank() as i32).unwrap();
            let all_r=comm.inner.all_gather_f64(&r);let all_y=comm.inner.all_gather_f64(&y);
            let m=n*np;let mut stacked=vec![0.;m*n];
            for rank in 0..np {for j in 0..n {for i in 0..n {
                stacked[rank*n+i+j*m]=all_r[rank*n*n+i+j*n];
            }}}
            let solution=K::least_squares(m,n,&stacked,&all_y);
            comm.close();self.coefficients=solution?;
            context.broadcast(0,&mut self.coefficients);
            self.diagnostics.tuned=true;
        }
        context.sum_f64(&mut self.totals);
        self.diagnostics.average_total_time=self.totals[0]/np as f64;
        self.diagnostics.average_over_time=self.totals[1]/np as f64;
        self.diagnostics.average_under_time=self.totals[2]/np as f64;
        self.totals=[0.;3];Ok(tuned)
    }
}

/// Source cube_params order: linear; i>=j quadratic; i>=j>=k cubic.
pub fn cubic_features(parameters:&[f64])->Vec<f64> {
    let n=parameters.len();let mut out=parameters.to_vec();
    let mut cubic=Vec::with_capacity(n*(n+1)*(n+2)/6);
    for i in 0..n {for j in 0..=i {
        let square=parameters[i]*parameters[j];out.push(square);
        for k in 0..=j {cubic.push(square*parameters[k]);}
    }}
    out.extend(cubic);out
}
pub struct CubicModel {parameters:usize,linear:LinearModel}
impl CubicModel {
    pub fn new(name:impl Into<String>,parameters:usize,coefficients:Vec<f64>,history_size:usize)->Self {
        assert_eq!(coefficients.len(),parameters+parameters*(parameters+1)/2+parameters*(parameters+1)*(parameters+2)/6);
        Self {parameters,linear:LinearModel::new(name,coefficients,history_size)}
    }
    pub fn linear(&self)->&LinearModel {&self.linear}
    pub fn estimate(&self,parameters:&[f64])->f64 {
        assert_eq!(parameters.len(),self.parameters);self.linear.estimate(&cubic_features(parameters))
    }
    pub fn observe(&mut self,seconds:f64,parameters:&[f64]) {
        assert_eq!(parameters.len(),self.parameters);self.linear.observe(seconds,&cubic_features(parameters));
    }
    pub fn update<K:LocalKernels>(&mut self,context:&Context<'_>)->Result<bool,i32> {self.linear.update::<K>(context)}
}
