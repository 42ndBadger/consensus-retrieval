use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use ahash::RandomState;

use crate::alias::AliasTable;

use rand;
use rand_distr::Distribution;

pub fn from_value_distribution<V>(n: usize, distibution: &HashMap<&V, f64>) -> HashMap<u64, V>
where
    V: Copy + Eq + Hash + Debug,
{
    let hasher = RandomState::new();
    let alias_table = AliasTable::new(distibution).unwrap();

    unique_keys_u64(n, &hasher)
        .iter()
        .map(|key| {
            let r = hasher.hash_one(*key);
            (*key, *alias_table.sample(r))
        })
        .collect()
}

pub fn from_distribution<V, D>(n: usize, distribution: D) -> HashMap<u64, V>
where
    V: Copy + Eq + Hash + Debug,
    D: Distribution<V>,
{
    unique_keys_u64(n, &RandomState::new())
        .into_iter()
        .zip(distribution.sample_iter(rand::rng()))
        .collect()
}

pub fn unique_keys_u64(n: usize, hasher: &RandomState) -> Vec<u64> {
    let mut result: HashSet<u64> = HashSet::new();

    for i in 0..n {
        let mut key = hasher.hash_one(i);
        let mut j = 0;
        while result.contains(&key) {
            key = hasher.hash_one(i + n * j);
            j += 1;
        }
        result.insert(key);
    }
    result.into_iter().collect()
}
