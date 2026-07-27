use crate::alias::{AliasTable, AliasTableError};
use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

pub type Seed = u64;
pub type HashCode = u64;

pub struct RetrievalHasher<K: Hash, V: Clone + Hash + Eq, H: BuildHasher> {
    _p: PhantomData<(K, V)>,
    alias_table: AliasTable<V>,
    build_hasher: H,
}

impl<K: Hash, V: Clone + Hash + Eq, H: BuildHasher> RetrievalHasher<K, V, H> {
    pub fn new_with_hasher(
        probabilities: &HashMap<&V, f64>,
        hash_builder: H,
    ) -> Result<Self, AliasTableError<V>> {
        let alias_table = AliasTable::from(probabilities)?;
        Ok(Self {
            _p: PhantomData,
            alias_table,
            build_hasher: hash_builder,
        })
    }
    
    /// Hashes `value` within its own `tag` domain, so that the different
    /// `hash_to_*`/`hash` methods never collide with each other just because
    /// they were called with the same underlying `key`.
    fn hash64(&self, tag: u8, value: impl Hash) -> u64 {
        let mut hasher = self.build_hasher.build_hasher();
        tag.hash(&mut hasher);
        value.hash(&mut hasher);
        hasher.finish()
    }

    pub fn hash_to_group(&self, key: HashCode, num_groups: usize) -> usize {
        fast_range(self.hash64(0, key), num_groups)
    }

    /// Must return "independent" hash values for different `num_tasks`.
    pub fn hash_to_task(&self, key: HashCode, num_tasks: usize) -> usize {
        // num_tasks is folded into the hashed bytes (not just the modulus),
        // so results for different num_tasks don't correlate.
        fast_range(self.hash64(1, (key, num_tasks)), num_tasks)
    }

    pub fn hash(&self, key: HashCode, seed: Seed) -> V {
        let h = self.hash64(2, (key, seed));
        self.alias_table.sample(h).clone()
    }

    pub fn hash_to_hash_code(&self, key: &K) -> HashCode {
        self.hash64(3, key)
    }

    /// Also checks wheter no two keys have the same hash code.
    /// Returns `None` if that's the case.
    pub fn convert_to_hash_codes(&self, kv: &HashMap<K, V>) -> Option<HashMap<HashCode, V>> {
        let mut result = HashMap::with_capacity(kv.len());
        for (k, v) in kv {
            let code = self.hash_to_hash_code(k);
            if result.insert(code, v.clone()).is_some() {
                return None;
            }
        }
        Some(result)
    }

    /// Total space (stack + heap) this structure occupies, in bytes.
    pub fn space_in_bytes(&self) -> usize {
        // `size_of::<Self>()` already covers `alias_table`'s own shallow
        // (stack) footprint (plus `build_hasher` and the zero-sized
        // `PhantomData`); add only its heap contents on top.
        size_of::<Self>()
            + (self.alias_table.space_in_bytes() - size_of_val(&self.alias_table))
    }
}

#[inline]
fn fast_range(seed: Seed, max: usize) -> usize {
    ((seed as u128 * max as u128) >> Seed::BITS) as usize
}
