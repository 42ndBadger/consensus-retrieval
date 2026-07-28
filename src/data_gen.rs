use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use ahash::RandomState;

use crate::alias::AliasTable;

pub fn from_value_distribution<V>(n: usize, distibution: &HashMap<&V, f64>) -> HashMap<u64, V>
where
    V: Copy + Eq + Hash + Debug,
{
    let hasher = RandomState::new();
    let mut result = HashMap::new();

    let alias_table = AliasTable::new(distibution).unwrap();
    for i in 0..n {
        let mut key = hasher.hash_one(i);
        let mut j = 0;
        while result.contains_key(&key) {
            key = hasher.hash_one(i + n * j);
            j += 1;
        }
        let r = hasher.hash_one(key);
        result.insert(key, *alias_table.sample(r));
    }
    result
}
