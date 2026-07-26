use std::{collections::HashMap, hash::Hash};
use sux::prelude::*;

use crate::{
    Probabilities,
    hasher::{HashCode, RetrievalHasher},
};

pub struct InsertionVec {
    num_groups: usize,
    select: SelectZeroAdapt<Rank9>,
    b: usize,
    β: usize,
}

impl InsertionVec {
    pub fn new<K: Hash, V: Clone + Eq + Hash>(
        kv: &HashMap<HashCode, V>,
        probabilities: &Probabilities<V>,
        b: usize,
        hasher: &RetrievalHasher<K, V>,
    ) -> Self {
        let β = f64::ceil((b as f64).sqrt() * (b as f64).log2()) as usize;
        let ε = 1. / (b as f64);
        #[allow(non_snake_case)]
        let H: f64 = probabilities.values().map(|p| -p * p.log2()).sum();
        let λ = (b as f64 + β as f64 / 2.) * (1. - ε) / H;

        let num_groups = ((kv.len() as f64) / λ).ceil() as usize;

        let mut kv_per_group = vec![vec![]; num_groups];
        for (key, value) in kv {
            let group = hasher.hash_to_group(*key, num_groups);
            kv_per_group[group].push((key, value));
        }

        let max_l = -ε.log2() + 3. * β as f64;

        // l_right = -log2(q_right); boundary q_right = 1 (vacuous success) => l = 0
        let mut l_right = 0f64;
        let mut num_insertions_per_group = vec![0usize; num_groups];
        for (group, group_kv) in kv_per_group.iter().enumerate().rev() {
            for num_insertions in 0.. {
                let mut good_event = true;
                let num_tasks = b + β * num_insertions;
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

                    if l > max_l {
                        good_event = false;
                        break;
                    }
                }

                if l > max_l - β as f64 {
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
        let select = SelectZeroAdapt::new(Rank9::new(bits));

        Self {
            num_groups,
            select,
            b,
            β,
        }
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// ell_i
    fn num_group_insertions(&self, group_idx: usize) -> Option<usize> {
        if group_idx >= self.num_groups {
            return None;
        }
        let end = self.select.select_zero(group_idx)?;
        let start = match group_idx {
            0 => 0,
            _ => self.select.select_zero(group_idx - 1)? + 1,
        };
        Some(end - start)
    }

    pub fn group_size(&self, group_idx: usize) -> Option<usize> {
        let num_ins = self.num_group_insertions(group_idx)?;
        Some(self.b + num_ins * self.β)
    }

    // in bits
    pub fn group_start(&self, group_idx: usize) -> Option<usize> {
        // TODO more efficient without iteration?
        if group_idx > self.num_groups {
            return None;
        }
        Some(
            (0..group_idx)
                .map(|g| self.b + self.β * self.group_size(g).expect("valid"))
                .sum(),
        )
    }

    pub fn total_num_tasks(&self) -> usize {
        self.group_start(self.num_groups).expect("valid")
    }
}
