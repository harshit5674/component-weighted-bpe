//! Probes on the final update of the constructed family (Figure 2 of the paper).
//!
//!     cargo run --release --example witness_probes -- [max_k] [table_budget_gib]
//!
//! For k = 4..=max_k (default 10), builds the dictionary with t = 2^k, runs
//! both branch searches over its input, checks that they agree on every byte,
//! and prints the interval probes on the final update next to the exact counts
//! k(k+1)/2 + 1 and k. It also reports the median time of the final update
//! over 31 runs after one warm-up run; timings depend on the machine.
//!
//! The total spelling length of these vocabularies grows as t², and so does
//! the dense matching automaton. Measured peak memory: about 1.5 GB for
//! k = 11 and 4 GB for k = 12; k = 13 needs roughly four times more. Single
//! appends take well under a microsecond here, close to the resolution of the
//! system clock, so the timings are coarse; the probe counts are exact.
use component_weighted_bpe::witness::{Letters, Witness};
use component_weighted_bpe::*;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let max_k: usize = args.first().map_or(Ok(10), |s| s.parse())?;
    let budget_gib: usize = args.get(1).map_or(Ok(8), |s| s.parse())?;
    println!("k,t,count_balanced_probes,weighted_probes,formula_count_balanced,formula_weighted,count_balanced_final_ns,weighted_final_ns");
    for k in 4..=max_k {
        let w = Witness::new(k, Letters::DistinctPairs)?;
        let mut probes = Vec::new();
        let mut times = Vec::new();
        let mut first: Vec<TokenId> = Vec::new();
        for (m, search) in [
            CentroidSearch::CountBalanced,
            CentroidSearch::ComponentWeighted,
        ]
        .into_iter()
        .enumerate()
        {
            let options = CentroidOptions {
                search,
                max_table_bytes: budget_gib << 30,
            };
            let compiled = Arc::new(CentroidCompiled::compile(w.dictionary.clone(), &options)?);
            let mut stream = CentroidStream::with_capacity(compiled, w.input.len())?;
            let mut stats = SearchStats::default();
            for (i, &b) in w.input.iter().enumerate() {
                let (token, s) = stream.append_profiled(b)?;
                stats = s;
                if m == 0 {
                    first.push(token);
                } else if token != first[i] {
                    return Err(format!("searches disagree at k={k}, byte {}", i + 1).into());
                }
            }
            probes.push(stats.interval_probes);
            // Time only the final append: rebuild the stream up to it each run.
            let (body, last) = w.input.split_at(w.input.len() - 1);
            let mut samples = Vec::new();
            for run in 0..32 {
                stream.reset();
                stream.extend(body)?;
                let start = Instant::now();
                let token = stream.append(last[0])?;
                let elapsed = start.elapsed().as_nanos();
                std::hint::black_box(token);
                if run > 0 {
                    samples.push(elapsed);
                }
            }
            samples.sort_unstable();
            times.push(samples[samples.len() / 2]);
        }
        let expected = (w.count_balanced_probes(), w.component_weighted_probes());
        println!(
            "{k},{},{},{},{},{},{},{}",
            w.max_token_length(),
            probes[0],
            probes[1],
            expected.0,
            expected.1,
            times[0],
            times[1]
        );
        if (probes[0], probes[1]) != expected {
            return Err(format!("probe counts differ from the formulas at k={k}").into());
        }
    }
    Ok(())
}
