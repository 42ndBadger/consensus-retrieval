use std::collections::HashMap;

use consensus_retrieval::{ConsensusRetrieval, parameters::Parameters};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn construct(c: &mut Criterion) {
    let n = 1_000;
    let sigma = 3;
    println!("n={n}, sigma={sigma}");

    let mut rng = StdRng::seed_from_u64(42);
    let kv: HashMap<_, _> = (0..n).map(|k| (k, rng.random_range(0..sigma))).collect();
    println!("11: {}", kv[&11u32]);
    let b = 40;
    let params = Parameters::new_from_raw(b, 1. / b as f64, 0.1);
    // let params = Parameters::new_like_in_proof(b);

    c.bench_function("construct_uniform_3_1000", |b| {
        b.iter(|| {
            ConsensusRetrieval::new_with_parameters(
                &kv,
                params,
                ahash::RandomState::with_seeds(
                    1123213325248739821,
                    1092830217302921830,
                    987213987219837321,
                    !1298372198372121322,
                ),
            )
        })
    });
}

criterion_group!(benches, construct);
criterion_main!(benches);
