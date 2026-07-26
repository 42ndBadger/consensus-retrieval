use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub enum AliasTableError<V> {
    EmptyInput,
    InvalidWeight(V, f64),
    ZeroTotalWeight,
}
struct Entry<V> {
    weight: f64,
    value_and_alias: [V; 2], // value, alias — in array to allow branchless selection
}
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
