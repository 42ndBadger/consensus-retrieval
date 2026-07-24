use std::{collections::HashMap, hash::Hash, marker::PhantomData};

use crate::{hasher::RetrievalHasher, insertion_vec::InsertionVec};

mod consensus;
mod insertion_vec;
mod hasher;


pub struct ConsensusRetrieval<K: Hash, V: Clone> {
   insertion_vec: insertion_vec::InsertionVec,
   consensus_vector: consensus::ConsensusVector,
   hasher: hasher::RetrievalHasher<K, V>,
}

impl<K: Hash, V: Clone> ConsensusRetrieval<K, V> {
    pub fn new(kv: HashMap<K, V>, group_size: u32) -> Self {
        // let mut frequencies = kv.values().collect::<Vec<&V>>();
        // frequencies.sort();
        
        // let hasher = RetrievalHasher::new_random(frequencies);
        // let insertion = InsertionVec::new(kv, group_size, hasher)
        todo!()
    }

    pub fn query(&self, key: &K) -> V {
       todo!() 
    }
}