use indicatif::ProgressIterator;
use mem_dbg::{MemSize, SizeFlags};
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    mem::size_of_val,
};
use sux::prelude::*;

use crate::{
    Probabilities,
    hasher::{HashCode, RetrievalHasher},
    parameters::Parameters,
};

pub struct InsertionVec {
    num_groups: usize,
    select: SelectZeroAdapt<Rank9>,
    b: usize,
    β: usize,
}

pub struct GroupBounds {
    pub start: usize,
    pub width: usize,
}

impl InsertionVec {
    pub fn new<K: Hash, V: Clone + Eq + Hash + Debug>(
        kv: &HashMap<HashCode, V>,
        probabilities: &Probabilities<V>,
        parms: Parameters,
        hasher: &RetrievalHasher<K, V, impl BuildHasher>,
    ) -> Self {
        let b = parms.inital_group_width;
        #[allow(non_snake_case)]
        let H: f64 = probabilities.values().map(|p| -p * p.log2()).sum();
        dbg!(&H);
        let λ = parms.avg_group_load / H;
        dbg!(&λ);

        let num_groups = ((kv.len() as f64) / λ).ceil() as usize;
        dbg!(&num_groups);

        let mut kv_per_group = vec![vec![]; num_groups];
        for (key, value) in kv {
            let group = hasher.hash_to_group(*key, num_groups);
            kv_per_group[group].push((key, value));
        }

        let max_keys_per_group: usize = kv_per_group.iter().map(Vec::len).max().unwrap();
        println!("max key per group {}", max_keys_per_group);

        // l_right = -log2(q_right); boundary q_right = 1 (vacuous success) => l = 0
        let mut l_right = 0f64;
        let mut num_insertions_per_group = vec![0usize; num_groups];
        for (group, group_kv) in kv_per_group.iter().enumerate().rev().progress() {
            for num_insertions in 0.. {
                let mut good_event = true;
                let num_tasks = b + parms.insertion_increment * num_insertions;
                let mut task_log_p = vec![0f64; num_tasks];

                for (key, value) in group_kv {
                    let task = hasher.hash_to_task(**key, num_tasks);
                    task_log_p[task] -= probabilities[*value].log2();
                }

                let mut l = l_right;
                for task in (0..num_tasks).rev() {
                    // q_i = 1 - (1 - q_(i+1) p_i)^2 = x(2-x) with x = q_(i+1) p_i
                    // l_i := -log2(q_i) = l_(i+1) + (-log2 p_i) - log2(2 - q_(i+1) p_i)
                    // l stays small (q close to 1), so exponentiating it back to
                    // recover q_(i+1)/p_i for the correction term doesn't underflow,
                    // unlike accumulating q_i itself which saturates to 1.0.
                    let q_next = (-l).exp2();
                    let p = (-task_log_p[task]).exp2();
                    l = l + task_log_p[task] - (2. - q_next * p).log2();

                    if l > parms.max_difficulty_of_task {
                        good_event = false;
                        break;
                    }
                }

                if l > parms.max_difficulty_at_group_border {
                    good_event = false;
                }

                if good_event {
                    num_insertions_per_group[group] = num_insertions;
                    l_right = l;
                    break;
                }
            }
        }

        // Unary code, group by group in order: ell_i ones followed by a 0 divider.
        let mut bits = BitVec::with_capacity(
            num_insertions_per_group.iter().sum::<usize>() + num_insertions_per_group.iter().len(),
        );
        for &ell in &num_insertions_per_group {
            for _ in 0..ell {
                bits.push(true);
            }
            bits.push(false);
        }
        // println!("insertion {bits}");
        let select = SelectZeroAdapt::new(Rank9::new(bits));

        Self {
            num_groups,
            select,
            b: parms.inital_group_width,
            β: parms.insertion_increment,
        }
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn group_bounds(&self, group_idx: usize) -> Option<GroupBounds> {
        if group_idx >= self.num_groups {
            return None;
        }

        let ins_end = self.select.select_zero(group_idx)?;
        let ins_start = match group_idx {
            0 => 0,
            _ => self.select.select_zero(group_idx - 1)? + 1,
        };
        let num_insertions = ins_end - ins_start;

        let group_width = self.b + num_insertions * self.β;

        let insertions_in_prior_groups = if group_idx > 0 {
            ins_start - group_idx
        } else {
            0
        };
        let group_start = group_idx * self.b + insertions_in_prior_groups * self.β;

        Some(GroupBounds {
            start: group_start,
            width: group_width,
        })
    }

    pub fn total_num_tasks(&self) -> usize {
        self.β * (self.select.len() - self.num_groups()) + self.b * self.num_groups()
    }

    /// Total space (stack + heap) this structure occupies, in bytes.
    pub fn space_in_bytes(&self) -> usize {
        let InsertionVec {
            num_groups,
            select,
            b,
            β,
        } = self;

        select.mem_size(SizeFlags::default())
            + size_of_val(num_groups)
            + size_of_val(b)
            + size_of_val(β)
    }

    pub fn variable_part_bit_size(&self) -> usize {
        self.select.mem_size(SizeFlags::default()) * u8::BITS as usize
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use crate::{
        hasher::{HashCode, RetrievalHasher},
        insertion_vec::InsertionVec,
        parameters::Parameters,
    };

    #[test]
    fn small_group_boundies_match() {
        let n = 100;
        let mut kv = HashMap::new();
        for i in 0..n {
            kv.insert(i, i % 2);
        }
        let probabilities: HashMap<&HashCode, f64> = HashMap::from([(&0, 0.5), (&1, 0.5)]);
        let params = Parameters::new_like_in_proof(2);
        let hasher: RetrievalHasher<u64, u64, ahash::RandomState> =
            RetrievalHasher::new_with_hasher(&probabilities, ahash::RandomState::new()).unwrap();

        let insertion_vec = InsertionVec::new(&kv, &probabilities, params, &hasher);

        // H = 1
        let num_groups = (n as f64 / params.avg_group_load).ceil() as usize;
        assert!(insertion_vec.num_groups() == num_groups);

        let mut expected_next_group_start = 0;
        for g in 0..num_groups {
            let bounds = insertion_vec.group_bounds(g).unwrap();
            assert!(
                bounds.start == expected_next_group_start,
                "expected group {g} to start at {expected_next_group_start} but it started at {}",
                bounds.start
            );
            assert!(bounds.width >= params.inital_group_width);
            expected_next_group_start += bounds.width;
        }

        assert!(expected_next_group_start == insertion_vec.total_num_tasks());
    }

    #[test]
    /// AI Generated
    fn max_difficulties_are_never_exceeded() {
        let n = 100;
        let mut kv = HashMap::new();
        for i in 0..n {
            kv.insert(i, 0 == i % 4);
        }
        let probabilities: HashMap<&bool, f64> = HashMap::from([(&true, 0.25), (&false, 0.75)]);
        let params = Parameters::new_like_in_proof(2);
        let hasher: RetrievalHasher<u64, bool, ahash::RandomState> =
            RetrievalHasher::new_with_hasher(&probabilities, ahash::RandomState::new()).unwrap();

        let insertion_vec = InsertionVec::new(&kv, &probabilities, params, &hasher);
        let num_groups = insertion_vec.num_groups();

        // Group items exactly like `InsertionVec::new` does, so we can replay
        // the same task assignment (hash_to_task is a pure function of
        // (key, num_tasks) for a given hasher) using the num_tasks that was
        // actually committed to for each group, and check that the l values
        // the construction is supposed to have bounded really are.
        let mut kv_per_group = vec![vec![]; num_groups];
        for (key, value) in &kv {
            let group = hasher.hash_to_group(*key, num_groups);
            kv_per_group[group].push((key, value));
        }

        let mut l_right = 0f64;
        for group in (0..num_groups).rev() {
            let bounds = insertion_vec.group_bounds(group).unwrap();
            let num_tasks = bounds.width;

            let mut task_log_p = vec![0f64; num_tasks];
            for (key, value) in &kv_per_group[group] {
                let task = hasher.hash_to_task(**key, num_tasks);
                task_log_p[task] -= probabilities[*value].log2();
            }

            let mut l = l_right;
            for task in (0..num_tasks).rev() {
                let q_next = (-l).exp2();
                let p = (-task_log_p[task]).exp2();
                l = l + task_log_p[task] - (2. - q_next * p).log2();

                assert!(
                    l <= params.max_difficulty_of_task,
                    "group {group} task {task}: l={l} exceeds max_difficulty_of_task={}",
                    params.max_difficulty_of_task
                );
            }

            assert!(
                l <= params.max_difficulty_at_group_border,
                "group {group}: border l={l} exceeds max_difficulty_at_group_border={}",
                params.max_difficulty_at_group_border
            );

            l_right = l;
        }
    }
}
