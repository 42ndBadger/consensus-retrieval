#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub avg_group_load: f64,        // λ * H
    pub inital_group_width: usize,  // b
    pub insertion_increment: usize, // β
    pub max_difficulty_of_task: f64,
    pub max_difficulty_at_group_border: f64,
}

impl Parameters {
    pub fn new_like_in_proof(b: usize) -> Self {
        assert!(b > 0, "b must be positive");
        let ε = 1. / (b as f64 + 1.); // todo how to ensure < 1?
        let β = f64::ceil((b as f64).sqrt() * (b as f64).log2()) as usize + 1; // todo how ensure > 0?
        let avg_group_load = (b as f64 + β as f64 / 2.) * (1. - ε);
        Self {
            avg_group_load,
            inital_group_width: b,
            insertion_increment: β,
            max_difficulty_of_task: -ε.log2() + 3. * β as f64,
            max_difficulty_at_group_border: -ε.log2() + 2. * β as f64
        }
    }
}
