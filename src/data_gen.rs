use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};

use ahash::RandomState;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use rand_distr::Distribution;

use crate::alias::AliasTable;

/// Builds a deterministic (hasher, sampling RNG) pair from a seed. The seed
/// is expanded via a CSPRNG into four keys for `RandomState::with_seeds`,
/// which expects high-quality random keys rather than a raw user-supplied
/// seed.
pub fn seeded_state(seed: u64) -> (RandomState, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let hasher = RandomState::with_seeds(rng.random(), rng.random(), rng.random(), rng.random());
    (hasher, StdRng::seed_from_u64(seed))
}

/// Samples `n` values from an explicit weight table via a seeded alias table.
/// Iteration order of the result is deterministic because both the result and
/// the `distribution` map are keyed by the fixed `hasher`.
pub fn from_value_distribution<V, H: BuildHasher + Clone>(
    n: usize,
    hasher: &H,
    distribution: &HashMap<&V, f64, H>,
) -> HashMap<u64, V, H>
where
    V: Copy + Eq + Hash + Debug,
{
    let alias_table = AliasTable::new(distribution).unwrap();
    let mut result = HashMap::with_capacity_and_hasher(n, hasher.clone());
    for key in unique_keys_u64(n, hasher) {
        let r = hasher.hash_one(key);
        result.insert(key, *alias_table.sample(r));
    }
    result
}

/// Samples `n` values from a `rand_distr` distribution using `rng`.
pub fn from_distribution<V, D, H: BuildHasher + Clone>(
    n: usize,
    hasher: &H,
    rng: impl Rng,
    distribution: D,
) -> HashMap<u64, V, H>
where
    V: Copy + Eq + Hash + Debug,
    D: Distribution<V>,
{
    let mut result = HashMap::with_capacity_and_hasher(n, hasher.clone());
    for (key, value) in unique_keys_u64(n, hasher)
        .into_iter()
        .zip(distribution.sample_iter(rng))
    {
        result.insert(key, value);
    }
    result
}

/// Generates `n` distinct `u64` keys by hashing an index. Iteration order of
/// the returned vector is deterministic because the internal `HashSet` uses
/// the fixed `hasher`.
pub fn unique_keys_u64<H: BuildHasher + Clone>(n: usize, hasher: &H) -> Vec<u64> {
    let mut result: HashSet<u64, H> = HashSet::with_hasher(hasher.clone());

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

#[cfg(test)]
mod seed_tests {
    use super::*;

    #[test]
    fn same_seed_is_deterministic() {
        let g = rand_distr::Geometric::new(0.5).unwrap();
        let (hasher, rng) = seeded_state(7);
        let a = from_distribution::<u64, _, _>(20, &hasher, rng, g);
        let (_, rng) = seeded_state(7);
        let b = from_distribution::<u64, _, _>(20, &hasher, rng, g);
        let mut a: Vec<_> = a.into_iter().collect();
        let mut b: Vec<_> = b.into_iter().collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn same_seed_value_dist_is_deterministic() {
        let (hasher, _) = seeded_state(7);
        let values = [0u64, 1, 2, 3, 4];
        let dist = weighted_dist(&hasher, &values, &[0.2; 5]);
        let mut a = from_value_distribution::<u64, _>(20, &hasher, &dist)
            .into_iter()
            .collect::<Vec<_>>();
        let mut b = from_value_distribution::<u64, _>(20, &hasher, &dist)
            .into_iter()
            .collect::<Vec<_>>();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn same_seed_multinomial_is_deterministic() {
        let (hasher, _) = seeded_state(7);
        let values = [0u64, 1, 2];
        let dist = weighted_dist(&hasher, &values, &[1.0, 2.0, 3.0]);
        let mut a = from_value_distribution::<u64, _>(20, &hasher, &dist)
            .into_iter()
            .collect::<Vec<_>>();
        let mut b = from_value_distribution::<u64, _>(20, &hasher, &dist)
            .into_iter()
            .collect::<Vec<_>>();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    fn weighted_dist<'a>(
        hasher: &RandomState,
        keys: &'a [u64],
        weights: &[f64],
    ) -> HashMap<&'a u64, f64, RandomState> {
        let mut dist: HashMap<&u64, f64, RandomState> = HashMap::with_hasher(hasher.clone());
        for (key, weight) in keys.iter().zip(weights) {
            dist.insert(key, *weight);
        }
        dist
    }
}
