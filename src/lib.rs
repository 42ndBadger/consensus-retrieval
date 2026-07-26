use std::{collections::HashMap, fmt::Debug, hash::Hash, marker::PhantomData};

use crate::{
    hasher::{HashCode, RetrievalHasher},
    insertion_vec::InsertionVec,
};

mod alias;
mod consensus;
mod hasher;
mod insertion_vec;

pub struct ConsensusRetrieval<K: Hash, V: Clone + Hash + Eq> {
    insertion_vec: insertion_vec::InsertionVec,
    consensus_vector: consensus::ConsensusVector,
    hasher: hasher::RetrievalHasher<K, V>,
}

type Probabilities<'a, V> = HashMap<&'a V, f64>;

impl<K: Hash, V: Clone + Hash + Eq + Debug> ConsensusRetrieval<K, V> {
    pub fn new(kv: HashMap<K, V>, group_size: usize) -> Self {
        let frequencies = calculate_frequencies(&kv);
        let hasher = RetrievalHasher::new_random(&frequencies).unwrap();
        let kv: HashMap<HashCode, V> = hasher.conert_to_hash_codes(&kv).expect("not duplicates");

        let insertion = InsertionVec::new(&kv, &frequencies, group_size, &hasher);
        let consensus = consensus::ConsensusVector::new(&kv, &insertion, &hasher);

        Self {
            insertion_vec: insertion,
            consensus_vector: consensus,
            hasher,
        }
    }

    pub fn query(&self, key: &K) -> V {
        let key = self.hasher.hash_to_hash_code(key);
        let group = self
            .hasher
            .hash_to_group(key, self.insertion_vec.num_groups());
        let group_size = self.insertion_vec.group_size(group).expect("valid group");
        let in_group_offset = self.hasher.hash_to_task(key, group_size);
        let group_start = self.insertion_vec.group_start(group).expect("vaid group");
        let consensus_idx = group_start + in_group_offset;
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
    use crate::calculate_frequencies;

    #[test]
    fn test_calc_frequencies() {
        let kv = [(0, 0), (1, 1), (2, 0), (3, 0)].into();
        let probs = calculate_frequencies(&kv);
        assert_eq!(probs, [(&1, 0.25), (&0, 0.75)].into());
    }
}
