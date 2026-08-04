use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
};

use fxhash::FxBuildHasher;

use crate::{
    hasher::{HashCode, RetrievalHasher},
    insertion_vec::InsertionVec,
    parameters::Parameters,
};

mod alias;
mod consensus;
pub mod data_gen;
pub mod hasher;
mod insertion_vec;
pub mod parameters;

pub struct ConsensusRetrieval<K: Hash, V: Clone + Hash + Eq, H: BuildHasher = FxBuildHasher>
where
    H::Hasher: Clone,
{
    insertion_vec: insertion_vec::InsertionVec,
    consensus_vector: consensus::ConsensusVector,
    hasher: hasher::RetrievalHasher<K, V, H>,
    entropy_per_key: f64,
    num_keys: usize,
}

type Probabilities<'a, V, S> = HashMap<&'a V, f64, S>;

impl<K: Hash, V: Clone + Hash + Eq + Debug> ConsensusRetrieval<K, V> {
    pub fn new_random(kv: &HashMap<K, V>, b: usize) -> Self {
        Self::new_with_hasher(kv, b, FxBuildHasher::new())
    }
}

impl<K: Hash, V: Clone + Hash + Eq + Debug, H: BuildHasher + Clone> ConsensusRetrieval<K, V, H>
where
    H::Hasher: Clone,
{
    pub fn new_with_hasher(
        kv: &HashMap<K, V, impl BuildHasher>,
        b: usize,
        hasher_bulder: H,
    ) -> Self {
        Self::new_with_parameters(kv, Parameters::new_like_in_proof(b), hasher_bulder)
    }

    pub fn new_with_parameters(
        kv: &HashMap<K, V, impl BuildHasher>,
        params: Parameters,
        hasher_bulder: H,
    ) -> Self {
        if kv.is_empty() {
            panic!("empty input")
        }
        let frequencies = calculate_frequencies(kv, hasher_bulder.clone());
        let hasher = RetrievalHasher::new_with_hasher(&frequencies, hasher_bulder).unwrap();
        let kv: HashMap<HashCode, V> = hasher.convert_to_hash_codes(kv).expect("not duplicates");

        dbg!(&params);
        let insertion = InsertionVec::new(&kv, &frequencies, params, &hasher);
        let consensus = consensus::ConsensusVector::new(&kv, &insertion, &hasher);

        let entropy_per_key: f64 = frequencies.values().map(|p| -p * p.log2()).sum();
        Self {
            insertion_vec: insertion,
            consensus_vector: consensus,
            hasher,
            entropy_per_key,
            num_keys: kv.iter().len(),
        }
    }

    /// Total space (stack + heap) this data structure occupies, in bytes.
    pub fn space_in_bytes(&self) -> usize {
        let ConsensusRetrieval {
            insertion_vec,
            consensus_vector,
            hasher,
            entropy_per_key,
            num_keys,
        } = self;
        insertion_vec.space_in_bytes()
            + consensus_vector.space_in_bytes()
            + hasher.space_in_bytes()
            + size_of_val(entropy_per_key)
            + size_of_val(num_keys)
    }

    // Number of bits of part that scales with input size.
    pub fn variable_part_bit_size(&self) -> usize {
        self.insertion_vec.variable_part_bit_size() + self.consensus_vector.variable_part_bit_size()
    }

    /// Bits used by the insertion vector (the unary-coded insertion counts
    /// plus its rank/select index), one component of `variable_part_bit_size`.
    pub fn insertion_vec_bit_size(&self) -> usize {
        self.insertion_vec.variable_part_bit_size()
    }
    /// Bit size of the insertion vector without the select index.
    pub fn raw_insertion_vec_bit_size(&self) -> usize {
        self.insertion_vec.raw_insertion_vec_bit_size()
    }

    /// Bits used by the consensus vector, the other component of
    /// `variable_part_bit_size`.
    pub fn consensus_vec_bit_size(&self) -> usize {
        self.consensus_vector.variable_part_bit_size()
    }

    pub fn num_tasks(&self) -> usize {
        self.insertion_vec.total_num_tasks()
    }

    pub fn hash_evaluations(&self) -> u64 {
        self.consensus_vector.hash_evaluations()
    }

    pub fn space_overhead(&self) -> f64 {
        self.variable_part_bit_size() as f64 / self.num_keys as f64 - self.entropy_per_key
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
        self.hasher.hash(key, seed).clone()
    }
}

fn calculate_frequencies<K: Hash, V: Clone + Hash + Eq, S: BuildHasher>(
    kv: &HashMap<K, V, impl BuildHasher>,
    hasher: S,
) -> Probabilities<'_, V, S> {
    let total_num = kv.len() as f64;
    let num_vals = kv.values().fold(HashMap::<_, usize>::new(), |mut acc, v| {
        *acc.entry(v).or_default() += 1;
        acc
    });
    let mut map = HashMap::with_hasher(hasher);
    map.extend(
        num_vals
            .into_iter()
            .map(|(v, num)| (v, num as f64 / total_num)),
    );
    map
}

#[cfg(test)]
mod test {
    use rand::random_range;

    use crate::{ConsensusRetrieval, calculate_frequencies};

    #[test]
    fn test_calc_frequencies() {
        let kv = [(0, 0), (1, 1), (2, 0), (3, 0)].into();
        let probs = calculate_frequencies(&kv, std::hash::RandomState::new());
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
