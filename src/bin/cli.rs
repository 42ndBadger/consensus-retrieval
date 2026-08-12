use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::{collections::HashMap, hash::BuildHasher};

use clap::Parser;
use consensus_retrieval::export_statistics;
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
    /// Only needed when actually building the retrieval structure, i.e.
    /// not when just generating data with --distribution --output.
    #[arg(short, required_unless_present = "output")]
    b: Option<usize>,
    #[arg(long, required_unless_present = "output")]
    beta_scale: Option<f64>,
    #[arg(long, required_unless_present = "output")]
    eps_scale: Option<f64>,

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
fn read_kv_file(path: &str) -> HashMap<String, u64> {
    let contents = fs::read_to_string(path).unwrap_or_else(|e| panic!("can't read {path:?}: {e}"));
    contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (key, value) = line
                .split_once(' ')
                .unwrap_or_else(|| panic!("expected `key value`, got {line:?}"));
            let value: u64 = value
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("invalid value {value:?}"));
            (key.to_string(), value)
        })
        .collect()
}

/// Writes `key value` pairs, one per line, in the same format `read_kv_file`
/// reads, so generated data can be saved and reloaded via `--file` later.
fn write_kv_file(path: &str, kv: &HashMap<String, u64>) {
    let mut contents = String::with_capacity(kv.len() * 8);
    for (key, value) in kv {
        contents.push_str(&key);
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
    /// probability distributions from `rand_distr`, sampled directly via a
    /// real RNG through `data_gen::from_distribution`.
    fn generate(self, n: usize) -> HashMap<u64, u64> {
        match self {
            Distribution::Uniform { count } => {
                let weight = 1.0 / count as f64;
                let weights: HashMap<u64, f64> = (0..count as u64).map(|i| (i, weight)).collect();
                let dist: HashMap<&u64, f64> = weights.iter().map(|(k, v)| (k, *v)).collect();
                data_gen::from_value_distribution(n, &dist)
            }
            Distribution::Multinomial { weights } => {
                let weights: HashMap<u64, f64> = weights
                    .into_iter()
                    .enumerate()
                    .map(|(i, w)| (i as u64, w))
                    .collect();
                let dist: HashMap<&u64, f64> = weights.iter().map(|(k, v)| (k, *v)).collect();
                data_gen::from_value_distribution(n, &dist)
            }
            Distribution::Binomial { trials, p } => {
                let binomial = rand_distr::Binomial::new(trials, p)
                    .unwrap_or_else(|e| panic!("invalid binomial distribution: {e}"));
                data_gen::from_distribution(n, binomial)
            }
            Distribution::Zipf { n: set_size, s } => {
                let zipf = rand_distr::Zipf::new(set_size, s)
                    .unwrap_or_else(|e| panic!("invalid zipf distribution: {e}"))
                    .map(|rank: f64| rank as u64);
                data_gen::from_distribution(n, zipf)
            }
            Distribution::Geometric { p } => {
                let geometric = rand_distr::Geometric::new(p)
                    .unwrap_or_else(|e| panic!("invalid geometric distribution: {e}"));
                data_gen::from_distribution(n, geometric)
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

    let kv = match Input::from_cli(cli.file, cli.distribution) {
        Input::File(path) => read_kv_file(&path),
        Input::Distribution(distribution) => {
            let n = cli
                .n
                .expect("clap guarantees -n is set when --distribution is used");
            let kv = distribution
                .generate(n)
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            if let Some(path) = &cli.output {
                // Just generating data to save for later: skip building the
                // (potentially expensive) retrieval structure entirely.
                write_kv_file(path, &kv);
                return;
            }
            kv
        }
    };

    let params = build_params(cli.b, cli.beta_scale, cli.eps_scale);

    if cli.stats_only {
        print_stats(&kv, params);
        return;
    }

    build_and_report(&kv, params);
}

/// Unwraps the three "only needed when actually building" args. clap
/// guarantees these are `Some` whenever this is reached, via
/// `required_unless_present = "output"` on each of them.
fn build_params(b: Option<usize>, beta_scale: Option<f64>, eps_scale: Option<f64>) -> Parameters {
    Parameters::new_with_scales(
        b.expect("clap guarantees -b is set when building"),
        beta_scale.expect("clap guarantees --beta-scale is set when building"),
        eps_scale.expect("clap guarantees --eps-scale is set when building"),
    )
}

fn build_and_report<K: Hash, V: Clone + Hash + Eq + Debug>(kv: &HashMap<K, V>, params: Parameters) {
    let cr = consensus_retrieval::ConsensusRetrieval::new_with_parameters(
        kv,
        params,
        ahash::RandomState::with_seed(123),
    );

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
) {
    let probabilities = &calculate_frequencies(kv, ahash::RandomState::new());
    let hasher =
        &RetrievalHasher::new_with_hasher(probabilities, ahash::RandomState::new()).unwrap();
    let kv_new = &hasher
        .convert_to_hash_codes(kv)
        .expect("no duplicate hash codes");
    let insertion_vec = InsertionVec::new(kv_new, probabilities, params, hasher);

    println!("===OUTPUT===");
    println!(
        r#"{{"tasks": {}, "bitvec_size": {}, "total_insertion_vec_size": {}}}"#,
        insertion_vec.total_num_tasks(),
        insertion_vec.bitvec_bits(),
        insertion_vec.variable_part_bit_size()
    );
    println!("key, value, group, task");
    for (key, value, group, task) in export_statistics(&insertion_vec, hasher, kv.iter()) {
        println!("{}, {}, {}, {}", key, value, group, task);
    }
}
