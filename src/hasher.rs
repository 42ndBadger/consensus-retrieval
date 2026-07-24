use std::{collections::HashMap, hash::Hash, marker::PhantomData};

pub type Seed = u64;
pub type HashCode = u64;

pub struct RetrievalHasher<K: Hash, V: Clone> {
    _p: PhantomData<(K, V)>,
}

impl<K: Hash, V: Clone> RetrievalHasher<K, V> {
    pub fn new_random(probabilities: &HashMap<&V, f64>) -> Self {
        todo!()
    }
    
    pub fn hash_to_group(&self, key: HashCode, num_groups: usize) -> usize {
        todo!()
    }

    /// Must return "independent" hash values for different `num_tasks`.
    pub fn hash_to_task(&self, key: HashCode, num_tasks: usize) -> usize {
        todo!()
    }

    pub fn hash(&self, key: HashCode, seed: Seed) -> V {
        todo!()
    }

    pub fn hash_to_hash_code(&self, key: &K) -> HashCode {
        todo!()
    }

    
    /// Also checks wheter no two keys have the same hash code.
    /// Returns `None` if that's the case.
    pub fn conert_to_hash_codes(&self, kv: &HashMap<K, V>) -> Option<HashMap<HashCode, V>> {
        todo!()
    }
}
