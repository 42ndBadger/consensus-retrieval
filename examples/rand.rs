use consensus_retrieval::ConsensusRetrieval;
use rand::random_range;

fn main() {
    let n = 80;
    let sigma = 3;
    let b = 5;

    let kv = (0..n).map(|k| (k, random_range(0..sigma))).collect();
    let retrieval = ConsensusRetrieval::new_random(&kv, b);

    for (k, v) in kv.iter() {
        assert!(
            &retrieval.query(k) == v,
            "expected {k} -> {v} but got {}",
            retrieval.query(k)
        )
    }
}
