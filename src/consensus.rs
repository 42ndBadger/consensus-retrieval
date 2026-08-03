use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    mem::size_of_val,
};

use indicatif::ProgressBar;
use mem_dbg::{MemSize, SizeFlags};
use sux::{
    bits::BitVec,
    traits::{BitCount, BitVecValueOps},
};

use crate::{
    hasher::{HashCode, RetrievalHasher, Seed},
    insertion_vec::InsertionVec,
};

pub struct ConsensusVector {
    bitvec: BitVec<Vec<Seed>>,
    hash_evaluations: u64,
}

impl ConsensusVector {
    pub fn new<K: Hash, V: Clone + Hash + Eq + Debug>(
        kv: &HashMap<HashCode, V>,
        insertion_vec: &InsertionVec,
        hasher: &RetrievalHasher<K, V, impl BuildHasher>,
    ) -> Self {
        let tasks = get_consensus_tasks(kv, insertion_vec, hasher);

        let max_keys_per_task = tasks.iter().map(Vec::len).max().unwrap();
        println!("max keys per task: {max_keys_per_task}");

        let progress = ProgressBar::new(tasks.len() as u64);
        let mut iterations: u64 = 0;
        let mut hash_evaluations: u64 = 0;
        // root seed has size Seed::BITS - 1
        let mut consensus_vec = BitVec::with_capacity(tasks.len() + Seed::BITS as usize - 1);
        // During the construction the consensus vector is consensus_vec with
        // current appended at the end
        let mut current: Seed = 0;

        println!("num_tasks {}", tasks.len());

        while consensus_vec.len() < tasks.len() {
            let task = consensus_vec.len();

            if iterations.is_multiple_of(1024) {
                progress.set_position(task as u64);
            }
            iterations += 1;

            let task_valid = tasks[task].iter().all(|&(k, v)| {
                hash_evaluations += 1;
                hasher.hash(k, current) == v
            });
            // println!(
            //     "Task {task} with keys {:?} is valid? {task_valid} seed {current}",
            //     tasks[task]
            // );

            if task_valid {
                // next task:
                // append a bit to the consensus vector by shifting the most
                // significant bit of current onto consusus_vec making space for
                // a new task in current
                let to_save = current >> (Seed::BITS - 1) == 1;
                // println!("saving {to_save}");
                consensus_vec.push(to_save);
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
        // fore some stupid reason, bits get added from right to left...
        consensus_vec.append_value(current.reverse_bits(), Seed::BITS as usize - 1);
        assert_eq!(consensus_vec.len(), tasks.len() + Seed::BITS as usize - 1);
        // println!("{consensus_vec}");
        println!(
            "ratio of ones: {}",
            consensus_vec.count_ones() as f32 / consensus_vec.len() as f32
        );
        Self {
            bitvec: consensus_vec,
            hash_evaluations,
        }
    }

    pub fn get_seed_at_task(&self, task_idx: usize) -> Seed {
        let seed = self
            .bitvec
            .get_value(task_idx, Seed::BITS as usize)
            .reverse_bits(); // todo avoid reverse...
        // println!("queried seed {seed} at {task_idx}");
        seed
    }

    /// Number of times `RetrievalHasher::hash` was called while constructing
    /// this consensus vector.
    pub fn hash_evaluations(&self) -> u64 {
        self.hash_evaluations
    }

    /// Total space (stack + heap) this structure occupies, in bytes.
    pub fn space_in_bytes(&self) -> usize {
        let ConsensusVector {
            bitvec,
            hash_evaluations,
        } = self;
        bitvec.mem_size(SizeFlags::default()) + size_of_val(hash_evaluations)
    }

    pub fn variable_part_bit_size(&self) -> usize {
        self.bitvec.len()
    }

    pub fn get_ratio_one_bits(&self) -> f32 {
        self.bitvec.count_ones() as f32 / self.bitvec.len() as f32
    }
}

fn get_consensus_tasks<'a, K: Hash, V: Clone + Hash + Eq + Debug>(
    kv: &'a HashMap<HashCode, V>,
    insertion_vec: &InsertionVec,
    hasher: &RetrievalHasher<K, V, impl BuildHasher>,
) -> Vec<Vec<(HashCode, &'a V)>> {
    let mut consensus_tasks = vec![Vec::new(); insertion_vec.total_num_tasks()];

    for (&k, v) in kv.iter() {
        let gidx = hasher.hash_to_group(k, insertion_vec.num_groups());
        let group_bounds = insertion_vec.group_bounds(gidx).unwrap();
        let offset = hasher.hash_to_task(k, group_bounds.width);
        consensus_tasks[group_bounds.start + offset].push((k, v));
    }

    consensus_tasks
}

#[cfg(test)]
mod test {
    use sux::bits::bit_vec;

    use crate::hasher::Seed;

    #[test]
    #[ignore]
    fn test_bitvec() {
        let mut bv = bit_vec![Seed];
        bv.append_value(0xF1, 8);
        println!("{bv}");
        panic!()
    }
}
