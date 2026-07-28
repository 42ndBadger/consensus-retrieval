use consensus_retrieval::{ConsensusRetrieval, parameters::Parameters};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn main() {
    let n = 1_000_000;
    let sigma = 3;
    println!("n={n}, sigma={sigma}");

    let mut rng = StdRng::seed_from_u64(42);
    let kv = (0..n).map(|k| (k, rng.random_range(0..sigma))).collect();

    let b = 10;
    let params = Parameters::new_from_raw(b, 1. / b as f64, 0.1);
    // let params = Parameters::new_like_in_proof(b);

    let retrieval =
        ConsensusRetrieval::new_with_parameters(&kv, params, ahash::RandomState::with_seed(11));
    println!("overhead {}", retrieval.space_overhead());
    println!("evals {}", retrieval.hash_evaluations());

    for (k, v) in kv.iter() {
        assert!(
            &retrieval.query(k) == v,
            "expected {k} -> {v} but got {}",
            retrieval.query(k)
        )
    }
}
