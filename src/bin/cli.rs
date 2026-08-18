use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::{collections::HashMap, hash::BuildHasher};

use ahash::RandomState;
use clap::Parser;
use consensus_retrieval::export_statistics;
use rand::rngs::StdRng;
use rand_distr::Distribution as _;

use consensus_retrieval::{
    calculate_frequencies, data_gen, hasher::RetrievalHasher, insertion_vec::InsertionVec,
    parameters::Parameters,
};

#[derive(clap::Parser)]
struct Cli {
    /// Number of items to generate; only used with --distribution.
    #[arg(short, required_unless_present = "file")]
    n: Option<usize>,

    #[clap(flatten)]
    algo_params: AlgoParams,

    /// Read `key value` pairs directly from a file (one pair per line,
    /// separated by a space), instead of generating them from a
    /// distribution.
    #[arg(
        long,
        conflicts_with = "distribution",
        required_unless_present = "distribution"
    )]
    file: Option<String>,

    /// Distribution to draw values from, followed by its parameters:
    /// `--distribution uniform <count>`,
    /// `--distribution multinomial <weight>...`,
    /// `--distribution binomial <trials> <p>`,
    /// `--distribution zipf <n> <s>`, or
    /// `--distribution geometric <p>`
    #[arg(
        long,
        short,
        num_args = 1..,
        value_name = "NAME [PARAMS...]",
        conflicts_with = "file",
        required_unless_present = "file"
    )]
    distribution: Option<Vec<String>>,

    /// Save the `key value` pairs generated from --distribution to this
    /// file, so they can be reloaded later with --file.
    #[arg(long)]
    output: Option<String>,

    /// Only print statistics about the generated data, not the data itself.
    #[arg(long, default_value = "false")]
    stats_only: bool,

    /// Seed for random data generation used. Defaults to a random seed.
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, clap::Args)]
struct AlgoParams {
    #[arg(short, required_unless_present = "output")]
    b: Option<usize>,
    #[arg(long, required_unless_present_any = ["output", "beta"])]
    beta_scale: Option<f64>,
    #[arg(long, required_unless_present_any = ["output", "eps"])]
    eps_scale: Option<f64>,

    #[arg(long, conflicts_with = "beta_scale")]
    beta: Option<usize>,
    // average group load in bits
    #[arg(long)]
    avg_group_load: Option<f64>,
    #[arg(long, conflicts_with = "eps_scale")]
    eps: Option<f64>,
    #[arg(long)]
    maxdiff: Option<f64>,
    #[arg(long)]
    maxdiff_boundry: Option<f64>,
}

/// Where the (key, value) data comes from: read directly from a file, or
/// synthesized by sampling from a named distribution. `Cli::file` and
/// `Cli::distribution` are mutually exclusive and one is required, so
/// exactly one variant here ever gets constructed.
enum Input {
    File(String),
    Distribution(Distribution),
}

impl Input {
    fn from_cli(file: Option<String>, distribution: Option<Vec<String>>) -> Self {
        match (file, distribution) {
            (Some(path), None) => Input::File(path),
            (None, Some(args)) => Input::Distribution(Distribution::parse(&args)),
            _ => unreachable!("clap guarantees exactly one of file/distribution is set"),
        }
    }
}

/// Reads `key value` pairs, one per line, separated by a space.
fn read_kv_file(path: &str, hasher: &RandomState) -> HashMap<String, u64, RandomState> {
    let contents = fs::read_to_string(path).unwrap_or_else(|e| panic!("can't read {path:?}: {e}"));
    let mut kv: HashMap<String, u64, RandomState> = HashMap::with_hasher(hasher.clone());
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        let (key, value) = line
            .split_once(' ')
            .unwrap_or_else(|| panic!("expected `key value`, got {line:?}"));
        let value: u64 = value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("invalid value {value:?}"));
        kv.insert(key.to_string(), value);
    }
    kv
}

/// Writes `key value` pairs, one per line, in the same format `read_kv_file`
/// reads, so generated data can be saved and reloaded via `--file` later.
fn write_kv_file(path: &str, kv: &HashMap<String, u64, RandomState>) {
    let mut contents = String::with_capacity(kv.len() * 8);
    for (key, value) in kv {
        contents.push_str(key);
        contents.push(' ');
        contents.push_str(&value.to_string());
        contents.push('\n');
    }
    fs::write(path, contents).unwrap_or_else(|e| panic!("can't write {path:?}: {e}"));
}

#[derive(Debug, Clone)]
enum Distribution {
    Uniform { count: usize },
    Multinomial { weights: Vec<f64> },
    Binomial { trials: u64, p: f64 },
    Zipf { n: f64, s: f64 },
    Geometric { p: f64 },
}

impl Distribution {
    fn parse(args: &[String]) -> Self {
        let (name, rest) = args
            .split_first()
            .expect("--distribution needs at least a distribution name");
        match name.as_str() {
            "uniform" => {
                let count = match rest {
                    [count] => count
                        .parse()
                        .unwrap_or_else(|_| panic!("invalid count {count:?}")),
                    [] => 2,
                    _ => panic!("uniform takes at most one parameter: the number of values"),
                };
                Distribution::Uniform { count }
            }
            "multinomial" => {
                let weights: Vec<f64> = rest
                    .iter()
                    .map(|w| w.parse().unwrap_or_else(|_| panic!("invalid weight {w:?}")))
                    .collect();
                assert!(!weights.is_empty(), "multinomial needs at least one weight");
                Distribution::Multinomial { weights }
            }
            "binomial" => {
                let [trials, p] = rest else {
                    panic!("binomial needs exactly two parameters: <trials> <p>");
                };
                let trials: u64 = trials
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid trial count {trials:?}"));
                let p: f64 = p.parse().unwrap_or_else(|_| panic!("invalid p {p:?}"));
                Distribution::Binomial { trials, p }
            }
            "zipf" => {
                let [n, s] = rest else {
                    panic!("zipf needs exactly two parameters: <n> <s>");
                };
                let n: f64 = n.parse().unwrap_or_else(|_| panic!("invalid n {n:?}"));
                let s: f64 = s.parse().unwrap_or_else(|_| panic!("invalid s {s:?}"));
                Distribution::Zipf { n, s }
            }
            "geometric" => {
                let [p] = rest else {
                    panic!("geometric needs exactly one parameter: <p>");
                };
                let p: f64 = p.parse().unwrap_or_else(|_| panic!("invalid p {p:?}"));
                Distribution::Geometric { p }
            }
            other => panic!(
                "unknown distribution {other:?}, expected `uniform`, `multinomial`, \
                 `binomial`, `zipf`, or `geometric`"
            ),
        }
    }

    /// Generates `n` (key, value) pairs by sampling values from this
    /// distribution. `uniform`/`multinomial` build an explicit weight table
    /// and go through our own alias-table sampling; the others are true
    /// probability distributions from `rand_distr`, sampled directly through
    /// `rng` via `data_gen::from_distribution`.
    fn generate(
        self,
        n: usize,
        hasher: &RandomState,
        rng: StdRng,
    ) -> HashMap<u64, u64, RandomState> {
        match self {
            Distribution::Uniform { count } => {
                let weight = 1.0 / count as f64;
                let mut weights: HashMap<u64, f64, RandomState> =
                    HashMap::with_hasher(hasher.clone());
                for i in 0..count as u64 {
                    weights.insert(i, weight);
                }
                let mut dist: HashMap<&u64, f64, RandomState> =
                    HashMap::with_hasher(hasher.clone());
                dist.extend(weights.iter().map(|(k, v)| (k, *v)));
                data_gen::from_value_distribution(n, hasher, &dist)
            }
            Distribution::Multinomial { weights } => {
                let mut weight_map: HashMap<u64, f64, RandomState> =
                    HashMap::with_hasher(hasher.clone());
                for (i, w) in weights.into_iter().enumerate() {
                    weight_map.insert(i as u64, w);
                }
                let mut dist: HashMap<&u64, f64, RandomState> =
                    HashMap::with_hasher(hasher.clone());
                dist.extend(weight_map.iter().map(|(k, v)| (k, *v)));
                data_gen::from_value_distribution(n, hasher, &dist)
            }
            Distribution::Binomial { trials, p } => {
                let binomial = rand_distr::Binomial::new(trials, p)
                    .unwrap_or_else(|e| panic!("invalid binomial distribution: {e}"));
                data_gen::from_distribution(n, hasher, rng, binomial)
            }
            Distribution::Zipf { n: set_size, s } => {
                let zipf = rand_distr::Zipf::new(set_size, s)
                    .unwrap_or_else(|e| panic!("invalid zipf distribution: {e}"))
                    .map(|rank: f64| rank as u64);
                data_gen::from_distribution(n, hasher, rng, zipf)
            }
            Distribution::Geometric { p } => {
                let geometric = rand_distr::Geometric::new(p)
                    .unwrap_or_else(|e| panic!("invalid geometric distribution: {e}"));
                data_gen::from_distribution(n, hasher, rng, geometric)
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    assert!(
        cli.output.is_none() || cli.distribution.is_some(),
        "--output can only be used together with --distribution"
    );

    let seed = cli.seed.unwrap_or_else(rand::random);
    let (hasher, sampling_rng) = data_gen::seeded_state(seed);

    let kv = match Input::from_cli(cli.file, cli.distribution) {
        Input::File(path) => read_kv_file(&path, &hasher),
        Input::Distribution(distribution) => {
            let n = cli
                .n
                .expect("clap guarantees -n is set when --distribution is used");
            let generated = distribution.generate(n, &hasher, sampling_rng);
            let mut kv: HashMap<String, u64, RandomState> = HashMap::with_hasher(hasher.clone());
            kv.extend(generated.into_iter().map(|(k, v)| (k.to_string(), v)));
            if let Some(path) = &cli.output {
                // Just generating data to save for later: skip building the
                // (potentially expensive) retrieval structure entirely.
                write_kv_file(path, &kv);
                return;
            }
            kv
        }
    };

    let params = build_params(&cli.algo_params);

    if cli.stats_only {
        print_stats(&kv, params, hasher);
        return;
    }

    build_and_report(&kv, params, hasher);
}

/// Unwraps the three "only needed when actually building" args. clap
/// guarantees these are `Some` whenever this is reached, via
/// `required_unless_present = "output"` on each of them.
fn build_params(param: &AlgoParams) -> Parameters {
    let b = param.b.expect("clap guarantees -b is set when building");

    let eps = param
        .eps
        .unwrap_or_else(|| param.eps_scale.unwrap() * 1. / (b as f64 + 1.));

    let beta = param.beta.unwrap_or_else(|| {
        (f64::ceil(param.beta_scale.unwrap() * (b as f64).sqrt() * (b as f64).log2()) as usize)
            .max(1)
    });
    let avg_group_load = param
        .avg_group_load
        .unwrap_or_else(|| (b as f64 + beta as f64 / 2.) * (1. - eps));
    let max_diff = param.maxdiff.unwrap_or(-eps.log2() + 3. * beta as f64);
    let max_diff_boundry = param.maxdiff_boundry.unwrap_or(max_diff - beta as f64);

    Parameters {
        avg_group_load,
        inital_group_width: b,
        insertion_increment: beta,
        max_difficulty_of_task: max_diff,
        max_difficulty_at_group_border: max_diff_boundry,
    }
}

fn build_and_report<K: Hash, V: Clone + Hash + Eq + Debug>(
    kv: &HashMap<K, V, impl BuildHasher>,
    params: Parameters,
    hasher_builder: RandomState,
) {
    let cr =
        consensus_retrieval::ConsensusRetrieval::new_with_parameters(kv, params, hasher_builder);

    let raw_insertion_vec_bits = cr.raw_insertion_vec_bit_size();
    let select_structure_bits = cr.insertion_vec_bit_size() - raw_insertion_vec_bits;

    println!(
        "num_keys: {}\nentropy_per_key [bit]: {}\nspace [byte]: {}\n\nspace overhead [bits/key]: {}\ninsertion_vec_bits [bit]: {}\nraw_insertion_vec_bits [bit]: {}\nselect_structure_bits [bit]: {}\nconsensus_vec_bits [bit]: {}\ntime [hash evaluations]: {}",
        cr.num_keys(),
        cr.entropy_per_key(),
        cr.space_in_bytes(),
        cr.space_overhead(),
        cr.insertion_vec_bit_size(),
        raw_insertion_vec_bits,
        select_structure_bits,
        cr.consensus_vec_bit_size(),
        cr.hash_evaluations()
    );
}

fn print_stats<K: Hash + Display, V: Clone + Hash + Eq + Debug + Display>(
    kv: &HashMap<K, V, impl BuildHasher>,
    params: Parameters,
    hasher_builder: RandomState,
) {
    // Reuses the same seeded `RandomState` `main` built via
    // `data_gen::seeded_state` (rather than `ahash::RandomState::new()` or
    // `::with_seed()`, both of which draw in per-process randomness — see
    // that function's doc comment) so `--seed` actually makes the
    // group/task assignment reproducible across runs, as documented.
    let probabilities = &calculate_frequencies(kv, hasher_builder.clone());
    let hasher = &RetrievalHasher::new_with_hasher(probabilities, hasher_builder).unwrap();
    let kv_new = &hasher
        .convert_to_hash_codes(kv)
        .expect("no duplicate hash codes");
    let insertion_vec = InsertionVec::new(kv_new, probabilities, params, hasher);

    let group_widths: Vec<usize> = (0..insertion_vec.num_groups())
        .map(|g| insertion_vec.group_bounds(g).unwrap().width)
        .collect();
    let group_widths_json = group_widths
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    println!("===OUTPUT===");
    println!(
        r#"{{"tasks": {}, "bitvec_size": {}, "total_insertion_vec_size": {}, "group_widths": [{}]}}"#,
        insertion_vec.total_num_tasks(),
        insertion_vec.bitvec_bits(),
        insertion_vec.variable_part_bit_size(),
        group_widths_json
    );
    println!("key, value, group, task");
    for (key, value, group, task) in export_statistics(&insertion_vec, hasher, kv.iter()) {
        println!("{}, {}, {}, {}", key, value, group, task);
    }
}
