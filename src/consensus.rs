use std::{collections::HashMap, hash::Hash};

use sux::{bits::BitVec, traits::BitVecValueOps};

use crate::{
    hasher::{HashCode, RetrievalHasher, Seed},
    insertion_vec::InsertionVec,
};

pub struct ConsensusVector {
    bitvec: BitVec<Vec<Seed>>,
}

impl ConsensusVector {
    pub fn new<K: Hash, V: Clone + Hash + Eq>(
        kv: &HashMap<HashCode, V>,
        insertion_vec: &InsertionVec,
        hasher: &RetrievalHasher<K, V>,
    ) -> Self {
        let tasks = get_consensus_tasks(kv, insertion_vec, hasher);
        let mut consensus_vec = BitVec::with_capacity(tasks.len() + Seed::BITS as usize - 1); // -1 ?
        let mut current: Seed = 0;

        while consensus_vec.len() < tasks.len() {
            let task = consensus_vec.len();
            if test_seed_valid(current, &tasks[task], hasher) {
                // next task
                consensus_vec.push(current >> (Seed::BITS - 1) == 1);
                current <<= 1;
                continue;
            }

            // invalid seed: backtrack to find next
            while current & 1 != 0 {
                // backtrack
                if let Some(bit) = consensus_vec.pop() {
                    current = current >> 1 | (bit as Seed) << (Seed::BITS - 1);
                } else {
                    // root seed
                    break;
                }
            }
            current += 1;
        }

        // final writeback
        consensus_vec.append_value(current >> 1, Seed::BITS as usize - 1);
        assert_eq!(consensus_vec.len(), tasks.len() + Seed::BITS as usize - 1);
        Self {
            bitvec: consensus_vec,
        }
    }

    pub fn get_seed_at_task(&self, task_idx: usize) -> Seed {
        self.bitvec.get_value(task_idx, Seed::BITS as usize)
    }
}

fn get_consensus_tasks<'a, K: Hash, V: Clone + Hash + Eq>(
    kv: &'a HashMap<HashCode, V>,
    insertion_vec: &InsertionVec,
    hasher: &RetrievalHasher<K, V>,
) -> Vec<Vec<(HashCode, &'a V)>> {
    let mut groups = vec![Vec::new(); insertion_vec.num_groups()];
    for (&k, v) in kv.iter() {
        let group = hasher.hash_to_group(k, insertion_vec.num_groups());
        groups[group].push((k, v));
    }
    // todo don't do work twice? reuse
    let mut consensus_tasks = vec![Vec::new(); insertion_vec.total_num_tasks()];
    for (gidx, group) in groups.iter().enumerate() {
        let group_start = insertion_vec.group_start(gidx).expect("valid");
        let num_tasks = insertion_vec.group_size(gidx).expect("valid");
        for &(k, v) in group {
            let offset = hasher.hash_to_task(k, num_tasks);
            consensus_tasks[group_start + offset].push((k, v));
        }
    }
    consensus_tasks
}

fn test_seed_valid<K: Hash, V: Clone + Hash + Eq>(
    seed: Seed,
    kv: &[(HashCode, &V)],
    hasher: &RetrievalHasher<K, V>,
) -> bool {
    kv.iter().all(|&(k, v)| &hasher.hash(k, seed) == v)
}
