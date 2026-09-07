//! Explicit tensor task recording and distributed subworld execution.
//! Scheduling flow follows `interface/schedule.cxx`; no global recorder or C++ AST.
use crate::{algebra::{CustomMonoid, Semiring, Wire}, context::Context,
    mapping::Distribution, tensor::Tensor};
use std::{collections::{BTreeMap, BTreeSet}, time::Instant};
#[path = "schedule_graph.rs"]
mod graph;
#[path = "schedule_partition.rs"]
mod partition;

type Action<'a, A> = dyn for<'c, 'r> FnMut(&mut BTreeMap<usize, Tensor<'c, 'r, A>>) + 'a;
struct Task<'a, A: Semiring> {
    tensors: BTreeSet<usize>,
    output: usize,
    action: Box<Action<'a, A>>,
}

/// Explicit recording of one-output operations on dense tensors identified by
/// caller-assigned IDs. Input IDs must include the output for read-modify-write.
/// Each action executes collectively inside its assigned child context.
pub struct Schedule<'a, A: Semiring> {
    tasks: Vec<Task<'a, A>>,
    costs: Vec<f64>,
    graph: graph::Graph,
    max_partitions: Option<usize>,
}

/// Seconds accumulated over execution waves, matching the source timer fields.
#[derive(Default, Debug)]
pub struct ScheduleTimer {
    pub comm_down_time: f64,
    pub exec_time: f64,
    pub imbalance_wall_time: f64,
    pub imbalance_accum_time: f64,
    pub comm_up_time: f64,
    pub total_time: f64,
}

impl<'a, A: Semiring + Clone> Schedule<'a, A> where A::Element: Wire {
    pub fn new(max_partitions: Option<usize>) -> Self {
        assert!(max_partitions != Some(0));
        Self { tasks: Vec::new(), costs: Vec::new(), graph: graph::Graph::default(), max_partitions }
    }

    /// Cost estimates are in seconds, supplied by the operation's existing
    /// planner. Recording is deterministic and must match on all parent ranks.
    pub fn add_operation<F>(&mut self, inputs: &[usize], output: usize, estimated_seconds: f64, action: F) -> usize
    where F: for<'c, 'r> FnMut(&mut BTreeMap<usize, Tensor<'c, 'r, A>>) + 'a {
        assert!(estimated_seconds.is_finite() && estimated_seconds > 0.0);
        let id = self.graph.add(inputs, output);
        self.tasks.push(Task { tensors: inputs.iter().copied().chain([output]).collect(), output, action: Box::new(action) });
        self.costs.push(estimated_seconds);
        id
    }

    /// Replay the recorded DAG. Parent tensors retain their distributions;
    /// temporary child tensors use the source's default cyclic mapping policy.
    pub fn execute(&mut self, context: &Context<'_>, tensors: &mut BTreeMap<usize, Tensor<'_, '_, A>>) -> ScheduleTimer {
        for task in &self.tasks { for id in &task.tensors {
            assert!(std::ptr::eq(tensors[id].context(), context));
        }}
        let mut run = self.graph.start();
        let mut timer = ScheduleTimer::default();
        while !run.ready.is_empty() {
            let total_start = Instant::now();
            let ready_costs: Vec<_> = run.ready.iter().map(|&id| self.costs[id]).collect();
            let assignments = partition::partition(&run.ready, &ready_costs, context.size(), self.max_partitions);
            let color = assignments.iter().position(|(_,ranks)| ranks.contains(&context.rank())).unwrap();
            let child = context.split(Some(color.try_into().unwrap()), context.rank().try_into().unwrap()).unwrap();
            let task_id = assignments[color].0;
            let mut local = BTreeMap::new();
            for &id in &self.tasks[task_id].tensors {
                let parent = &tensors[&id];
                local.insert(id, Tensor::new(&child, Distribution::cyclic(parent.distribution().shape.clone(),child.size()),parent.algebra().clone()));
            }
            let down_start = Instant::now();
            for (partition, (task, ranks)) in assignments.iter().enumerate() {
                for &id in &self.tasks[*task].tensors {
                    let parent = &tensors[&id];
                    let distribution = Distribution::cyclic(parent.distribution().shape.clone(),ranks.len());
                    let destination = if partition == color { Some(local.get_mut(&id).unwrap()) } else { None };
                    parent.add_to_subworld(destination, &distribution, parent.algebra().one(),parent.algebra().zero());
                }
            }
            timer.comm_down_time += down_start.elapsed().as_secs_f64();
            context.barrier();
            let exec_start = Instant::now();
            (self.tasks[task_id].action)(&mut local);
            let local_exec = exec_start.elapsed().as_secs_f64();
            context.barrier();
            timer.exec_time += exec_start.elapsed().as_secs_f64();
            let min = context.all_reduce(&CustomMonoid { identity: f64::INFINITY, addition: |a:&f64,b:&f64| a.min(*b) }, &local_exec);
            let max = context.all_reduce(&CustomMonoid { identity: 0.0, addition: |a:&f64,b:&f64| a.max(*b) }, &local_exec);
            let mut imbalance = [local_exec-min]; context.sum_f64(&mut imbalance);
            timer.imbalance_wall_time += max-min;
            timer.imbalance_accum_time += imbalance[0];
            let up_start = Instant::now();
            for (partition, (task, ranks)) in assignments.iter().enumerate() {
                let id = self.tasks[*task].output;
                let parent = tensors.get_mut(&id).unwrap();
                let distribution = Distribution::cyclic(parent.distribution().shape.clone(),ranks.len());
                let source = if partition == color { Some(&local[&id]) } else { None };
                parent.add_from_subworld(source,&distribution,parent.algebra().one(),parent.algebra().zero());
            }
            timer.comm_up_time += up_start.elapsed().as_secs_f64();
            drop(local);
            child.close();
            run.finish(&self.graph,&assignments.iter().map(|(task,_)|*task).collect::<Vec<_>>());
            timer.total_time += total_start.elapsed().as_secs_f64();
        }
        timer
    }
}
