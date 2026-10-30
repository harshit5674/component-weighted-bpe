//! Probe statistics and update times on a real vocabulary (Section 5 and
//! Appendix E of the paper).
//!
//!     cargo run --release --example vocab_stats -- VOCAB.ibpe TEXT... [--audits N] [--repeats R]
//!
//! VOCAB.ibpe comes from `scripts/prepare_vocab.py`. Each text is tokenized as
//! one continuous byte string, without pre-tokenization, normalization or
//! special tokens. For each text and each branch search this prints one JSON
//! line with:
//!
//! * `searched_pct`: updates whose search read at least one interval;
//! * `max_list`: the longest interval list searched;
//! * `probes` and `endpoint_comparisons`: p50/p95/p99/p99.9/max per update,
//!   over the updates that searched;
//! * `ns_per_update`: median over R timed passes (default 5) after one warm-up;
//! * `audits`: prefixes, evenly spaced and including the whole text, whose
//!   reconstructed tokenization was compared with offline heap BPE.
//!
//! Both searches must return the same token after every byte; the program
//! stops with an error otherwise. Timings depend on the machine.
use component_weighted_bpe::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Reads the IBPE1 format: magic, alphabet length (u32 LE), alphabet bytes,
/// rule count (u32 LE), then (left, right) pairs of u32 LE token IDs.
fn load(path: &Path) -> Result<Dictionary> {
    let data = std::fs::read(path)?;
    if !data.starts_with(b"IBPE1") || data.len() < 13 {
        return Err("invalid IBPE file".into());
    }
    let n = u32::from_le_bytes(data[5..9].try_into()?) as usize;
    if n == 0 || n > 256 || data.len() < 13 + n {
        return Err("invalid alphabet".into());
    }
    let alphabet = &data[9..9 + n];
    let r = u32::from_le_bytes(data[9 + n..13 + n].try_into()?) as usize;
    if data.len() != 13 + n + r * 8 {
        return Err("invalid merge payload".into());
    }
    let merges: Vec<(u32, u32)> = data[13 + n..]
        .chunks_exact(8)
        .map(|p| {
            (
                u32::from_le_bytes(p[..4].try_into().unwrap()),
                u32::from_le_bytes(p[4..].try_into().unwrap()),
            )
        })
        .collect();
    Ok(Dictionary::new(alphabet, &merges, 1 << 30)?)
}

/// Nearest-rank percentiles p50, p95, p99, p99.9 and the maximum.
fn tails(mut values: Vec<usize>) -> String {
    if values.is_empty() {
        return "null".into();
    }
    values.sort_unstable();
    let rank =
        |q: f64| values[((q * values.len() as f64).ceil() as usize).clamp(1, values.len()) - 1];
    format!(
        "[{},{},{},{},{}]",
        rank(0.50),
        rank(0.95),
        rank(0.99),
        rank(0.999),
        values[values.len() - 1]
    )
}

fn name(path: &Path) -> String {
    path.file_stem()
        .map_or(String::new(), |s| s.to_string_lossy().into_owned())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut paths, mut audits, mut repeats) = (Vec::new(), 8usize, 5usize);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--audits" => audits = args.next().ok_or("--audits needs a value")?.parse()?,
            "--repeats" => repeats = args.next().ok_or("--repeats needs a value")?.parse()?,
            _ => paths.push(arg),
        }
    }
    if paths.len() < 2 || repeats == 0 {
        return Err("usage: vocab_stats VOCAB.ibpe TEXT... [--audits N] [--repeats R]".into());
    }
    let vocab_path = Path::new(&paths[0]);
    let dictionary = Arc::new(load(vocab_path)?);
    let modes = [
        CentroidSearch::CountBalanced,
        CentroidSearch::ComponentWeighted,
    ];
    let mut compiled = Vec::new();
    for search in modes {
        let options = CentroidOptions {
            search,
            max_table_bytes: 4 << 30,
        };
        let start = Instant::now();
        let c = Arc::new(CentroidCompiled::compile(dictionary.clone(), &options)?);
        let r = c.report();
        println!(
            "{{\"kind\":\"compile\",\"vocab\":\"{}\",\"search\":\"{search:?}\",\"ms\":{:.2},\"tokens\":{},\"max_token_length\":{},\"centroid_nodes\":{},\"downward_intervals\":{},\"max_downward_degree\":{},\"charged_table_bytes\":{}}}",
            name(vocab_path),
            start.elapsed().as_secs_f64() * 1e3,
            dictionary.token_count(),
            r.max_token_length,
            r.centroid_nodes,
            r.downward_intervals,
            r.max_downward_degree,
            r.charged_table_bytes
        );
        compiled.push(c);
    }
    for text_path in paths[1..].iter().map(Path::new) {
        let text = std::fs::read(text_path)?;
        if let Some(&b) = text.iter().find(|&&b| dictionary.atom(b).is_none()) {
            return Err(
                format!("{}: byte {b} is not in the vocabulary", text_path.display()).into(),
            );
        }
        let mut reference: Vec<TokenId> = Vec::with_capacity(text.len());
        for (m, c) in compiled.iter().enumerate() {
            let mut stream = CentroidStream::with_capacity(c.clone(), text.len())?;
            let (mut probes, mut comparisons, mut max_list) = (Vec::new(), Vec::new(), 0);
            for (i, &b) in text.iter().enumerate() {
                let (token, s) = stream.append_profiled(b)?;
                if m == 0 {
                    reference.push(token);
                } else if token != reference[i] {
                    return Err(format!("searches disagree at byte {}", i + 1).into());
                }
                if s.interval_probes > 0 {
                    probes.push(s.interval_probes);
                    comparisons.push(s.endpoint_comparisons);
                    max_list = max_list.max(s.largest_branch_list);
                }
            }
            let mut checked = 0;
            for a in 1..=audits {
                let endpoint = text.len() * a / audits;
                if stream.reconstruct(endpoint)? != dictionary.tokenize_heap(&text[..endpoint])? {
                    return Err(format!("prefix {endpoint} differs from offline BPE").into());
                }
                checked += 1;
            }
            let mut samples = Vec::new();
            for run in 0..=repeats {
                stream.reset();
                let start = Instant::now();
                for &b in &text {
                    std::hint::black_box(stream.append(b)?);
                }
                if run > 0 {
                    samples.push(start.elapsed().as_nanos() as f64 / text.len().max(1) as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{{\"kind\":\"text\",\"vocab\":\"{}\",\"text\":\"{}\",\"search\":\"{:?}\",\"bytes\":{},\"searched_pct\":{:.2},\"max_list\":{},\"probes\":{},\"endpoint_comparisons\":{},\"ns_per_update\":{:.2},\"audits\":{}}}",
                name(vocab_path),
                name(text_path),
                c.report().search,
                text.len(),
                100.0 * probes.len() as f64 / text.len().max(1) as f64,
                max_list,
                tails(probes),
                tails(comparisons),
                samples[samples.len() / 2],
                checked
            );
        }
    }
    Ok(())
}
