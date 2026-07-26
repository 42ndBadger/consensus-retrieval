use std::{collections::HashMap, hash::Hash};

use crate::{
    hasher::{HashCode, RetrievalHasher, Seed},
    insertion_vec::InsertionVec,
};

pub struct ConsensusVector {}

impl ConsensusVector {
    pub fn new<K: Hash, V: Clone>(
        kv: &HashMap<HashCode, V>,
        insertion_vec: &InsertionVec,
        hasher: &RetrievalHasher<K, V>,
    ) -> Self {
        todo!()
    }

    pub fn get_seed_at_task(&self, task_idx: usize) -> Seed {
        todo!()
    }
}
