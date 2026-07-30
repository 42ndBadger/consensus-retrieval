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
        Self::new_from_raw(b, ε, 1.)
    }

    pub fn new_from_raw(b: usize, ε: f64, beta_scale: f64) -> Self {
        assert!(b > 0, "b must be positive: b={b}");
        assert!(ε > 0. && ε < 1., "ε must be strictly in 0..1: ε={ε}");

        let β = (f64::ceil(beta_scale * (b as f64).sqrt() * (b as f64).log2()) as usize).max(1);
        let avg_group_load = (b as f64 + β as f64 / 2.) * (1. - ε);
        Self {
            avg_group_load,
            inital_group_width: b,
            insertion_increment: β,
            max_difficulty_of_task: -ε.log2() + 3. * β as f64,
            max_difficulty_at_group_border: -ε.log2() + 2. * β as f64,
        }
    }

    pub fn new_from_raw_difficulty(b: usize, ε: f64, beta_scale: f64, max_diff: f64) -> Self {
        let β = (f64::ceil(beta_scale * (b as f64).sqrt() * (b as f64).log2()) as usize).max(1);
       Self {
           avg_group_load: (b as f64 + β as f64 / 2.) * (1. - ε ),
           inital_group_width: b,
           insertion_increment: β,
           max_difficulty_of_task: max_diff,
           max_difficulty_at_group_border: max_diff,
       } 
    }
}
