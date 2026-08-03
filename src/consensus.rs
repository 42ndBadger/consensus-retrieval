use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    mem::size_of_val,
};

use crate::{
    hasher::{HashCode, RetrievalHasher, Seed},
    insertion_vec::InsertionVec,
};
use bitvec::{field::BitField, order::Msb0, vec::BitVec};
use indicatif::ProgressBar;
use mem_dbg::{MemSize, SizeFlags};

pub struct ConsensusVector {
    bitvec: BitVec<Seed, Msb0>,
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
        let mut consensus_vec = ConsensusVecManager::new(tasks.len() + Seed::BITS as usize - 1);
        // During the construction the consensus vector is consensus_vec with
        // current appended at the end
        let mut current: Seed = 0;

        println!("num_tasks {}", tasks.len());

        while consensus_vec.current_task_idx() < tasks.len() {
            let task = consensus_vec.current_task_idx();

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
                // let to_save = current >> (Seed::BITS - 1) == 1;
                // println!("saving {to_save}");
                consensus_vec.rotate_in(&mut current);
                continue;
            }

            // invalid seed: backtrack to find next
            while current & 1 != 0 {
                // backtrack
                if consensus_vec.rotate_out(&mut current) {
                    break;
                }
            }
            current += 1;
        }

        // final writeback
        // fore some stupid reason, bits get added from right to left...
        let consensus_vec = consensus_vec.dismantle(current);
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
        let seed = self.bitvec[task_idx..][..Seed::BITS as usize].load_be();
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
        // bitvec.leng(SizeFlags::default()) + size_of_val(hash_evaluations)
        todo!()
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

struct ConsensusVecManager {
    consensus_vec: BitVec<Seed, Msb0>,
    cache: Seed,
    /// number of valid bits in the cache (from LSB)
    cache_valid: u32,
    /// number of bits not yet written to the cache (from LSB)
    cache_unwritten: u32,
}

impl ConsensusVecManager {
    pub fn new(cap: usize) -> Self {
        Self {
            consensus_vec: BitVec::with_capacity(cap),
            cache: 0,
            cache_valid: 0,
            cache_unwritten: 0,
        }
    }

    pub fn current_task_idx(&self) -> usize {
        self.consensus_vec.len() + self.cache_unwritten as usize
    }

    pub fn rotate_in(&mut self, current: &mut Seed) {
        if self.cache_unwritten == Seed::BITS {
            // println!("writeback");
            // write back
            self.consensus_vec
                .extend_from_bitslice(&BitVec::<_, Msb0>::from_element(self.cache));
            self.cache_unwritten = 0;
        }

        let mut cache_seed = (self.cache as u128) << Seed::BITS | *current as u128;
        cache_seed <<= 1;
        self.cache = (cache_seed >> Seed::BITS) as Seed;
        *current = cache_seed as Seed;

        self.cache_valid = self.cache_valid.saturating_add(1).min(Seed::BITS); // just add
        self.cache_unwritten = self.cache_unwritten.saturating_add(1).min(Seed::BITS); // just add

        assert!(self.cache_unwritten <= self.cache_valid);
    }

    /// Returns `true` if the root seed is reached.
    pub fn rotate_out(&mut self, current: &mut Seed) -> bool {
        if self.cache_valid == 0 {
            if self.consensus_vec.is_empty() {
                // no action, root seed, we can just increment later
                return true;
            }
            // println!("load");
            // load
            let to_read = self.consensus_vec.len().min(Seed::BITS as usize);
            self.cache = self
                .consensus_vec
                .split_off(self.consensus_vec.len() - to_read)
                .load_be();
            self.cache_valid = to_read as u32;
            self.cache_unwritten = self.cache_valid;
        }

        let mut cache_current = (self.cache as u128) << Seed::BITS | *current as u128;
        cache_current >>= 1;
        self.cache = (cache_current >> Seed::BITS) as Seed;
        *current = cache_current as Seed;

        // pop of last value
        self.consensus_vec.resize(
            self.consensus_vec
                .len()
                .saturating_sub((self.cache_unwritten == 0) as usize), // may be empty
            false,
        );
        self.cache_valid = self.cache_valid.saturating_sub(1);
        self.cache_unwritten = self.cache_unwritten.saturating_sub(1);
        assert!(self.cache_unwritten <= self.cache_valid);
        false
    }

    pub fn dismantle(mut self, final_seed: Seed) -> BitVec<Seed, Msb0> {
        self.consensus_vec.extend_from_bitslice(
            &BitVec::<_, Msb0>::from_element(self.cache)
                [(Seed::BITS - self.cache_unwritten) as usize..],
        );
        self.consensus_vec.extend_from_bitslice(
            &BitVec::<_, Msb0>::from_element(final_seed)[..Seed::BITS as usize - 1],
        );
        self.consensus_vec
    }
}

#[cfg(test)]
mod test {
    use bitvec::{bitvec, order::Msb0, vec::BitVec};
    use sux::bits::bit_vec;

    use crate::{consensus::ConsensusVecManager, hasher::Seed};

    #[test]
    #[ignore]
    fn test_bitvec() {
        let mut bv = bit_vec![Seed];
        bv.append_value(0xF1, 8);
        println!("{bv}");
        panic!()
    }

    #[test]
    #[ignore]
    fn test_bitvec2() {
        let mut bv = bitvec![Seed, Msb0; 0; 0];
        bv.push(true);
        bv.push(false);
        bv.extend_from_bitslice(
            &BitVec::<_, Msb0>::from_element(0x7usize)[..Seed::BITS as usize - 1],
        );
        println!("{bv}");
        println!("{}", bv.len());
        println!("{}", bv.len());
        panic!()
    }

    #[test]
    fn test_cache() {
        let mut cache = ConsensusVecManager::new(100);
        let mut val: Seed = Seed::MAX;
        cache.rotate_in(&mut val);
        println!("{val}");
        cache.rotate_in(&mut val);
        println!("{val}");
        for i in 0..64 {
            cache.rotate_in(&mut val);
        }
        println!("{val}");
        cache.rotate_in(&mut val);
        println!("{val}");
        val = 0;
        cache.rotate_out(&mut val);
        println!("{val}");
        val = 11;
        let vec = cache.dismantle(val);
        println!("{vec}");
        println!("{}", vec.len());
        panic!()
    }
}
