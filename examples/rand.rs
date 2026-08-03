use std::{collections::HashMap, time::Instant};

use consensus_retrieval::{ConsensusRetrieval, parameters::Parameters};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn main() {
    let n = 100_000;
    let sigma = 3;
    println!("n={n}, sigma={sigma}");

    let mut rng = StdRng::seed_from_u64(42);
    let kv: HashMap<u32, u32> = (0..n).map(|k| (k, rng.random_range(0..sigma))).collect();

    let b = 40;
    let params = Parameters::new_from_raw(b, 1. / b as f64, 0.1);
    // let params = Parameters::new_like_in_proof(b);

    let start = Instant::now();
    let retrieval = ConsensusRetrieval::new_with_parameters(
        &kv,
        params,
        ahash::RandomState::with_seeds(
            1123213325248739821,
            1092830217302921830,
            987213987219837321,
            !1298372198372121322,
        ),
    );
    let took = start.elapsed();
    println!("construction took {took:?}, {:?} per key", took / n);
    println!("overhead {}", retrieval.space_overhead());
    println!(
        "consensus size {}",
        human_bytes::human_bytes(retrieval.consensus_vec_bit_size() as f64 / 8.)
    );
    println!(
        "insertion size {}",
        human_bytes::human_bytes(retrieval.insertion_vec_bit_size() as f64 / 8.)
    );
    // println!(
    //     "space usage {}",
    //     human_bytes::human_bytes(retrieval.space_in_bytes() as f64)
    // );
    println!("evals {}", retrieval.hash_evaluations());

    for (k, v) in kv.iter() {
        assert!(
            &retrieval.query(k) == v,
            "expected {k} -> {v} but got {}",
            retrieval.query(k)
        )
    }
}
