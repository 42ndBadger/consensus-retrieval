use std::collections::HashMap;

use consensus_retrieval::hasher::RetrievalHasher;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fxhash::FxBuildHasher;
use rand::random;

fn hash(c: &mut Criterion) {
    let probabilities = [(0, 0.5), (1, 0.2), (2, 0.2), (3, 0.1)];
    let state = ahash::RandomState::with_seeds(
        1123213325248739821,
        1092830217302921830,
        987213987219837321,
        !1298372198372121322,
    );
    let state = FxBuildHasher::new();
    // let state = fxhash::FxBuildHasher::new();
    let mut probs = HashMap::with_hasher(state.clone());
    probs.extend(probabilities.iter().map(|(k, v)| (k, *v)));
    let hasher = RetrievalHasher::<u32, _, _>::new_with_hasher(&probs, state).expect("valid probs");

    let param = (1, 12321412);
    c.bench_with_input(
        BenchmarkId::new("hash fixed", format!("{:?}", param)),
        &param,
        |b, v| b.iter(|| hasher.hash(v.0, v.1)),
    );

    c.bench_function("hash rand", |b| {
        b.iter_batched(
            || (random(), random()),
            |(v, s)| hasher.hash(v, s),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, hash);
criterion_main!(benches);
