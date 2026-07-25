use std::{collections::HashMap, hash::Hash, fmt::Debug, marker::PhantomData};

use crate::{
    hasher::{HashCode, RetrievalHasher},
    insertion_vec::InsertionVec,
};

mod alias;
mod consensus;
mod hasher;
mod insertion_vec;

pub struct ConsensusRetrieval<K: Hash, V: Clone> {
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
        todo!()
    }
}

fn calculate_frequencies<K: Hash, V: Clone>(kv: &HashMap<K, V>) -> Probabilities<'_, V> {
    todo!()
}
