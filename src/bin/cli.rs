use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::hash::Hash;

use clap::Parser;
use statrs::distribution::{Binomial, Discrete};

use consensus_retrieval::{data_gen, parameters::Parameters};

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
    /// `--distribution multinomial <weight>...`, or
    /// `--distribution binomial <trials> <p>`
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
    let contents =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("can't read {path:?}: {e}"));
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
fn write_kv_file(path: &str, kv: &HashMap<u64, usize>) {
    let mut contents = String::with_capacity(kv.len() * 8);
    for (key, value) in kv {
        contents.push_str(&key.to_string());
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
            other => panic!(
                "unknown distribution {other:?}, expected `uniform`, `multinomial`, or `binomial`"
            ),
        }
    }

    fn into_weights(self) -> HashMap<usize, f64> {
        match self {
            Distribution::Uniform { count } => {
                let weight = 1.0 / count as f64;
                (0..count).map(|i| (i, weight)).collect()
            }
            Distribution::Multinomial { weights } => weights.into_iter().enumerate().collect(),
            Distribution::Binomial { trials, p } => {
                let binomial = Binomial::new(p, trials)
                    .unwrap_or_else(|e| panic!("invalid binomial distribution: {e}"));
                (0..=trials)
                    .map(|k| (k as usize, binomial.pmf(k)))
                    .collect()
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

    match Input::from_cli(cli.file, cli.distribution) {
        Input::File(path) => {
            let params = build_params(cli.b, cli.beta_scale, cli.eps_scale);
            build_and_report(&read_kv_file(&path), params);
        }
        Input::Distribution(distribution) => {
            let weights = distribution.into_weights();
            let dist: HashMap<&usize, f64> = weights.iter().map(|(k, v)| (k, *v)).collect();
            let n = cli.n.expect("clap guarantees -n is set when --distribution is used");
            let kv = data_gen::from_value_distribution(n, &dist);
            if let Some(path) = &cli.output {
                // Just generating data to save for later: skip building the
                // (potentially expensive) retrieval structure entirely.
                write_kv_file(path, &kv);
                return;
            }
            let params = build_params(cli.b, cli.beta_scale, cli.eps_scale);
            build_and_report(&kv, params);
        }
    }
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

    println!(
        "space: {}\n\nspace overhead: {}\ninsertion_vec_bits: {}\nconsensus_vec_bits: {}\ntime: {}",
        cr.space_in_bytes(),
        cr.space_overhead(),
        cr.insertion_vec_bit_size(),
        cr.consensus_vec_bit_size(),
        cr.hash_evaluations()
    );
}
