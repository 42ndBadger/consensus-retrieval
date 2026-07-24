use std::{collections::HashMap, hash::Hash};

use crate::hasher::{HashCode, RetrievalHasher};


pub struct InsertionVec {
    num_groups: u32,
    
}

impl InsertionVec {
    pub fn new<K: Hash, V: Clone>(kv: HashMap<HashCode, V>, group_size: u32, hasher: &RetrievalHasher<K, V>) -> Self {
        todo!("construct insertion vector by calculating q_i")
    }

    pub fn num_groups(&self) -> u32 {
        self.num_groups
    }
    
    /// ell_i
    pub fn num_group_insertions(&self, group_idx: u32) -> Option<u32> {
        todo!()
    }
    
}
