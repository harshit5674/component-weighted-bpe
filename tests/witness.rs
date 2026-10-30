use component_weighted_bpe::witness::{Letters, Witness};
use component_weighted_bpe::*;
use std::sync::Arc;

const MODES: [CentroidSearch; 2] = [
    CentroidSearch::CountBalanced,
    CentroidSearch::ComponentWeighted,
];

/// Runs both searches over the whole input, checking that they agree at every
/// byte, and returns each mode's statistics for the final byte together with
/// the final tokenization.
fn run(w: &Witness, every_prefix: bool) -> Vec<(SearchStats, Vec<TokenId>)> {
    let mut results = Vec::new();
    let mut first: Vec<TokenId> = Vec::new();
    for (m, mode) in MODES.into_iter().enumerate() {
        let options = CentroidOptions {
            search: mode,
            max_table_bytes: 1 << 30,
        };
        let compiled = Arc::new(CentroidCompiled::compile(w.dictionary.clone(), &options).unwrap());
        let mut stream = CentroidStream::with_capacity(compiled, w.input.len()).unwrap();
        let mut stats = SearchStats::default();
        for (i, &b) in w.input.iter().enumerate() {
            let (token, s) = stream.append_profiled(b).unwrap();
            stats = s;
            if m == 0 {
                first.push(token);
            } else {
                assert_eq!(token, first[i], "k={} endpoint={}", w.k, i + 1);
            }
            if every_prefix {
                let oracle = w.dictionary.tokenize_reference(&w.input[..=i]).unwrap();
                assert_eq!(stream.reconstruct(i + 1).unwrap(), oracle);
            }
        }
        results.push((stats, stream.reconstruct(w.input.len()).unwrap()));
    }
    results
}

fn assert_formulas(w: &Witness, results: &[(SearchStats, Vec<TokenId>)]) {
    let (count, weighted) = (&results[0].0, &results[1].0);
    let k = w.k;
    assert_eq!(count.interval_probes, k * (k + 1) / 2 + 1, "k={k}");
    assert_eq!(count.interval_probes, w.count_balanced_probes());
    assert_eq!(weighted.interval_probes, k, "k={k}");
    assert_eq!(weighted.interval_probes, w.component_weighted_probes());
    // The search visits the roots of T_k, ..., T_1 and ends with one failed branch search.
    for s in [count, weighted] {
        assert_eq!(s.centroid_visits, k);
        assert_eq!(s.downward_moves, k - 1);
        assert_eq!(s.terminal_misses, 1);
        assert_eq!(s.upward_moves, 0);
        assert!(!s.longest_suffix_fast_path);
        assert_eq!(s.largest_branch_list, 1 << (k - 1));
    }
    // The input tokenizes as CONTEXT+a_0 followed by S without a_0.
    let last = &results[0].1;
    assert_eq!(last, &results[1].1);
    assert_eq!(last.len(), 2);
    assert_eq!(w.dictionary.spelling(last[0]).unwrap(), &w.input[..2]);
    assert_eq!(w.dictionary.spelling(last[1]).unwrap(), &w.input[2..]);
    assert_eq!(last, &w.dictionary.tokenize_heap(&w.input).unwrap());
}

#[test]
fn distinct_letters_match_literal_bpe_and_the_probe_formulas() {
    for k in 1..=7 {
        let w = Witness::new(k, Letters::Distinct).unwrap();
        assert_eq!(w.max_token_length(), 1 << k);
        let results = run(&w, true);
        assert_formulas(&w, &results);
    }
}

#[test]
fn repeated_letters_with_distinct_pairs_match_the_probe_formulas() {
    for k in 1..=9 {
        let w = Witness::new(k, Letters::DistinctPairs).unwrap();
        let results = run(&w, k <= 6);
        assert_formulas(&w, &results);
    }
}

#[test]
fn distinct_letters_are_limited_to_k_7() {
    assert!(matches!(
        Witness::new(8, Letters::Distinct),
        Err(Error::Limit { .. })
    ));
    assert!(matches!(
        Witness::new(0, Letters::DistinctPairs),
        Err(Error::Limit { .. })
    ));
}
