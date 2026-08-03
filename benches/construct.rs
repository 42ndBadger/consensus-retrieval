use consensus_retrieval::{ConsensusRetrieval, parameters::Parameters};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn construct(c: &mut Criterion) {
    let n = 1_000;
    let sigma = 3;
    println!("n={n}, sigma={sigma}");

    let mut rng = StdRng::seed_from_u64(42);
    let kv = (0..n).map(|k| (k, rng.random_range(0..sigma))).collect();
    let b = 40;
    let params = Parameters::new_from_raw(b, 1. / b as f64, 0.1);
    // let params = Parameters::new_like_in_proof(b);

    c.bench_function("construct_uniform_3_1000", |b| {
        b.iter(|| {
            ConsensusRetrieval::new_with_parameters(&kv, params, ahash::RandomState::with_seed(11))
        })
    });
}

criterion_group!(benches, construct);
criterion_main!(benches);
