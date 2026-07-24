use std::{collections::HashMap, hash::Hash, marker::PhantomData};

pub type Seed = u64;
pub type HashCode = u64;

pub struct RetrievalHasher<K: Hash, V: Clone> {
    _p: PhantomData<(K, V)>,
}

impl<K: Hash, V: Clone> RetrievalHasher<K, V> {
    pub fn new_random(probabilities: HashMap<K, f32>) -> Self {
        todo!()
    }
    
    pub fn hash_to_group(&self, key: HashCode, num_partitions: u64) -> Seed {
        todo!()
    }

    /// Must return "independent" hash values for different `num_tasks`.
    pub fn hash_to_task(&self, key: HashCode, num_tasks: u64) -> Seed {
        todo!()
    }

    pub fn hash(&self, key: HashCode, seed: Seed) -> V {
        todo!()
    }

    pub fn hash_to_hash_code(&self, key: &K) -> HashCode {
        todo!()
    }

    /// Checks wheter no two keys have the same hash code.
    pub fn check_hash_codes_unique<'a>(
        &self,
        hash_codes: impl IntoIterator<Item = &'a K> + 'a,
    ) -> bool
    where
        K: 'a,
    {
        todo!()
    }
}
