use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub enum AliasTableError<V> {
    EmptyInput,
    InvalidWeight(V, f64),
    ZeroTotalWeight,
}
#[derive(Debug, Clone, PartialEq)]
struct Entry<V> {
    weight: f64,
    value_and_alias: [V; 2], // value, alias — in array to allow branchless selection
}
#[derive(Debug, Clone, PartialEq)]
pub struct AliasTable<V> {
    table: Vec<Entry<V>>,
    average_weight: f64,
}

impl<V: Clone + Hash + Eq> AliasTable<V> {
    /// Builds an [`AliasTable`] from a [`HashMap`] of values and their weights.
    /// Adapted from Algorthim 2 of "Parallel Weighted Random Sampling" by Hübschle-Schneider und Sanders
    /// http://arxiv.org/abs/1903.00227
    pub fn from(values_with_weights: &HashMap<&V, f64>) -> Result<Self, AliasTableError<V>> {
        let n = values_with_weights.len();
        if n == 0 {
            return Err(AliasTableError::EmptyInput);
        }

        let mut total_weight = 0f64;
        let mut table = Vec::with_capacity(n);
        for (&v, &w) in values_with_weights {
            if w < 0.0 {
                return Err(AliasTableError::InvalidWeight(v.clone(), w));
            }
            total_weight += w;
            table.push(Entry {
                weight: w,
                value_and_alias: [v.clone(), v.clone()],
            })
        }

        if total_weight == 0.0 {
            return Err(AliasTableError::ZeroTotalWeight);
        }
        let avg_weight = total_weight / n as f64;

        fn next_light<V>(table: &[Entry<V>], avg_weight: f64, from: usize) -> usize {
            (from..table.len())
                .find(|&i| table[i].weight <= avg_weight)
                .unwrap_or(table.len())
        }
        fn next_heavy<V>(table: &[Entry<V>], avg_weight: f64, from: usize) -> usize {
            (from..table.len())
                .find(|&i| table[i].weight > avg_weight)
                .unwrap_or(table.len())
        }

        let mut j = next_heavy(&table, avg_weight, 0);
        let mut i = next_light(&table, avg_weight, 0);

        if j == n {
            // All have weight = avg_weight
            return Ok(Self {
                table,
                average_weight: avg_weight,
            });
        }

        let mut w = table[j].weight;

        while j < n {
            if i < n && w > avg_weight {
                // Pack light bucket
                table[i].value_and_alias[1] = table[j].value_and_alias[0].clone();
                w -= avg_weight - table[i].weight;
                i = next_light(&table, avg_weight, i + 1);
            } else {
                // Pack heavy bucket
                table[j].weight = w;
                let j_prime = next_heavy(&table, avg_weight, j + 1);
                if j_prime == n {
                    break;
                }
                table[j].value_and_alias[1] = table[j_prime].value_and_alias[0].clone();
                w = table[j_prime].weight - (avg_weight - w);
                j = j_prime;
            }
        }

        Ok(Self {
            table,
            average_weight: avg_weight,
        })
    }
}

impl<V> AliasTable<V> {
    /// Given a uniform random number `u` in [0, 1), it return a value from the
    /// table with the probailities specified during construction. Runs in O(1)
    /// time.
    pub fn sample(&self, u: f64) -> &V {
        let n = self.table.len();
        let scaled = u * n as f64;
        let index = scaled as usize;
        let f = (scaled - index as f64) * self.average_weight;
        let entry = &self.table[index];

        &entry.value_and_alias[(f <= entry.weight) as usize]
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use super::*;

    #[test]
    fn test_uniform_alias_table() {
        let uniform_distribution =
            HashMap::from([(&1, 0.25f64), (&2, 0.25f64), (&3, 0.25f64), (&4, 0.25f64)]);
        let table = AliasTable::from(&uniform_distribution).unwrap();
        for i in 0..100 {
            let u = 1. / 100. * i as f64;
            let sample = table.sample(u);
            assert!(uniform_distribution.contains_key(sample));
        }
    }

    #[test]
    fn test_uniform_alias_table_matches() {
        let uniform_distribution =
            HashMap::from([(&1, 0.25f64), (&2, 0.25f64), (&3, 0.25f64), (&4, 0.25f64)]);
        let table = AliasTable::from(&uniform_distribution).unwrap();
        assert!(alias_table_matches(&table, &uniform_distribution));
    }

    #[test]
    fn test_non_uniform_alias_table_matches() {
        let non_uniform_distribution =
            HashMap::from([(&1, 0.1f64), (&2, 0.2f64), (&3, 0.3f64), (&4, 0.4f64)]);
        let table = AliasTable::from(&non_uniform_distribution).unwrap();
        assert!(alias_table_matches(&table, &non_uniform_distribution));
    }

    fn probabilites_of_alias_table<V: Clone + Hash + Eq + Debug>(
        alias_table: &AliasTable<V>,
    ) -> HashMap<&V, f64> {
        let mut result = HashMap::new();
        let prob_of_entry = 1. / alias_table.table.len() as f64;
        for entry in &alias_table.table {
            println!(
                "entry: {entry:?} p: {}",
                entry.weight / alias_table.average_weight / alias_table.table.len() as f64
            );
            let p = entry.weight / alias_table.average_weight * prob_of_entry;
            result
                .entry(&entry.value_and_alias[0])
                .and_modify(|w| *w += p)
                .or_insert(p);
            result
                .entry(&entry.value_and_alias[1])
                .and_modify(|w| *w += prob_of_entry - p)
                .or_insert(prob_of_entry - p);
        }
        result
    }

    fn alias_table_matches<V: Clone + Hash + Eq + Debug>(
        alias_table: &AliasTable<V>,
        expected: &HashMap<&V, f64>,
    ) -> bool {
        let actual = probabilites_of_alias_table(alias_table);
        println!("table: {alias_table:?}");
        println!("actual: {actual:?}, expected: {expected:?}");
        assert!(
            actual.len() == expected.len(),
            "number of keys does not match"
        );
        for (k, p) in expected {
            assert!(actual.contains_key(k), "key {k:?} is missing");
            assert!(
                (*p - actual[k]).abs() < f64::EPSILON,
                "probability of {k:?} does not match, expected {p}, got {}",
                actual[k]
            );
        }
        true
    }
}
