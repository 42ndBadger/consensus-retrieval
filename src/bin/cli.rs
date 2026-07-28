use std::collections::HashMap;

use clap::Parser;
use statrs::distribution::{Binomial, Discrete};

use consensus_retrieval::data_gen;

#[derive(clap::Parser)]
struct Cli {
    #[arg(short)]
    n: usize,

    /// Distribution to draw values from, followed by its parameters:
    /// `--distribution uniform <count>`,
    /// `--distribution multinomial <weight>...`, or
    /// `--distribution binomial <trials> <p>`
    #[arg(long, short, num_args = 1.., value_name = "NAME [PARAMS...]")]
    distribution: Vec<String>,
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
    let weights = Distribution::parse(&cli.distribution).into_weights();
    let dist: HashMap<&usize, f64> = weights.iter().map(|(k, v)| (k, *v)).collect();

    let kv = data_gen::from_value_distribution(cli.n, &dist);

    let cr = consensus_retrieval::ConsensusRetrieval::new_random(&kv, 5);

    let entropy = cli.n as f64 * dist.values().map(|v| -v * v.log2()).sum::<f64>();

    println!(
        "space: {}\n\nentropy: {}\nspace overhead: {}\ntime: {}",
        cr.space_in_bytes(),
        entropy,
        cr.space_in_bytes() as f64 / entropy,
        cr.hash_evaluations()
    );
}
