use std::collections::HashMap;

use consensus_retrieval::{ConsensusRetrieval, parameters::Parameters};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, random_range, rngs::StdRng};

fn query(c: &mut Criterion) {
    let n = 100_000;
    let sigma = 3;
    println!("n={n}, sigma={sigma}");

    let mut rng = StdRng::seed_from_u64(42);
    let kv: HashMap<u32, u32> = (0..n).map(|k| (k, rng.random_range(0..sigma))).collect();

    let b = 40;
    let params = Parameters::new_from_raw(b, 1. / b as f64, 0.1);
    // let params = Parameters::new_like_in_proof(b);

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

    let param = 28123;
    c.bench_with_input(BenchmarkId::new("query const", param), &param, |b, v| {
        b.iter(|| retrieval.query(v))
    });

    c.bench_function("query rand", |b| {
        b.iter_batched(
            || random_range(0..n),
            |v| retrieval.query(&v),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, query);
criterion_main!(benches);
