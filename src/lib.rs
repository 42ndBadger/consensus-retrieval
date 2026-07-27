use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    mem::{size_of, size_of_val},
};

use crate::{
    hasher::{HashCode, RetrievalHasher},
    insertion_vec::InsertionVec,
    parameters::Parameters,
};

mod alias;
mod consensus;
mod hasher;
mod insertion_vec;
mod parameters;

pub struct ConsensusRetrieval<K: Hash, V: Clone + Hash + Eq, H: BuildHasher = ahash::RandomState> {
    insertion_vec: insertion_vec::InsertionVec,
    consensus_vector: consensus::ConsensusVector,
    hasher: hasher::RetrievalHasher<K, V, H>,
}

type Probabilities<'a, V> = HashMap<&'a V, f64>;

impl<K: Hash, V: Clone + Hash + Eq + Debug> ConsensusRetrieval<K, V> {
    pub fn new_random(kv: &HashMap<K, V>, b: usize) -> Self {
        Self::new_with_hasher(kv, b, ahash::RandomState::new())
    }
}

impl<K: Hash, V: Clone + Hash + Eq + Debug, H: BuildHasher> ConsensusRetrieval<K, V, H> {
    pub fn new_with_hasher(kv: &HashMap<K, V>, b: usize, hasher_bulder: H) -> Self {
        if kv.is_empty() {
            panic!("empty input")
        }
        let frequencies = calculate_frequencies(kv);
        let hasher = RetrievalHasher::new_with_hasher(&frequencies, hasher_bulder).unwrap();
        let kv: HashMap<HashCode, V> = hasher.convert_to_hash_codes(kv).expect("not duplicates");

        let parameters = Parameters::new_like_in_proof(b);
        dbg!(&parameters);
        let insertion = InsertionVec::new(&kv, &frequencies, parameters, &hasher);
        let consensus = consensus::ConsensusVector::new(&kv, &insertion, &hasher);

        Self {
            insertion_vec: insertion,
            consensus_vector: consensus,
            hasher,
        }
    }

    /// Total space (stack + heap) this data structure occupies, in bytes.
    pub fn space_in_bytes(&self) -> usize {
        // `size_of::<Self>()` already covers each field's own shallow
        // (stack) footprint; add only their heap contents on top.
        size_of::<Self>()
            + (self.insertion_vec.space_in_bytes() - size_of_val(&self.insertion_vec))
            + (self.consensus_vector.space_in_bytes() - size_of_val(&self.consensus_vector))
            + (self.hasher.space_in_bytes() - size_of_val(&self.hasher))
    }

    pub fn hash_evaluations(&self) -> u64 {
        self.consensus_vector.hash_evaluations()
    }

    pub fn query(&self, key: &K) -> V {
        let key = self.hasher.hash_to_hash_code(key);
        let group = self
            .hasher
            .hash_to_group(key, self.insertion_vec.num_groups());
        let group_bounds = self.insertion_vec.group_bounds(group).unwrap();
        let in_group_offset = self.hasher.hash_to_task(key, group_bounds.width);
        let consensus_idx = group_bounds.start + in_group_offset;
        let seed = self.consensus_vector.get_seed_at_task(consensus_idx);
        self.hasher.hash(key, seed)
    }
}

fn calculate_frequencies<K: Hash, V: Clone + Hash + Eq>(
    kv: &HashMap<K, V>,
) -> Probabilities<'_, V> {
    let total_num = kv.len() as f64;
    let num_vals = kv.values().fold(HashMap::<_, usize>::new(), |mut acc, v| {
        *acc.entry(v).or_default() += 1;
        acc
    });
    num_vals
        .into_iter()
        .map(|(v, num)| (v, num as f64 / total_num))
        .collect()
}

#[cfg(test)]
mod test {
    use rand::random_range;

    use crate::{ConsensusRetrieval, calculate_frequencies};

    #[test]
    fn test_calc_frequencies() {
        let kv = [(0, 0), (1, 1), (2, 0), (3, 0)].into();
        let probs = calculate_frequencies(&kv);
        assert_eq!(probs, [(&1, 0.25), (&0, 0.75)].into());
    }

    #[test]
    fn test_small() {
        let kv = [(0, 0), (1, 1), (2, 0), (3, 0)].into();
        let ret = ConsensusRetrieval::new_random(&kv, 1);
        for (k, v) in kv.iter() {
            assert!(
                &ret.query(k) == v,
                "expected {k} -> {v} but got {}",
                ret.query(k)
            )
        }
    }

    #[test]
    #[ignore]
    fn test_rand() {
        let n = 80;
        let sigma = 3;
        let b = 5;

        let kv = (0..n).map(|k| (k, random_range(0..sigma))).collect();
        let retrieval = ConsensusRetrieval::new_random(&kv, b);
        for (k, v) in kv.iter() {
            assert!(
                &retrieval.query(k) == v,
                "expected {k} -> {v} but got {}",
                retrieval.query(k)
            )
        }
    }
}
