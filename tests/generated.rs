use component_weighted_bpe::*;
use std::sync::Arc;
mod support;
use support::{literal, rng};

const MODES: [CentroidSearch; 2] = [
    CentroidSearch::CountBalanced,
    CentroidSearch::ComponentWeighted,
];

fn compile(dict: Arc<Dictionary>, search: CentroidSearch) -> Arc<CentroidCompiled> {
    let options = CentroidOptions {
        search,
        ..Default::default()
    };
    Arc::new(CentroidCompiled::compile(dict, &options).unwrap())
}

/// Runs both searches over `bytes`, checking every prefix against literal BPE,
/// and returns each mode's statistics for the final byte.
fn check(dict: Arc<Dictionary>, bytes: &[u8]) -> Vec<SearchStats> {
    let mut final_stats = Vec::new();
    for mode in MODES {
        let compiled = compile(dict.clone(), mode);
        let mut profiled = CentroidStream::with_capacity(compiled.clone(), bytes.len()).unwrap();
        let mut plain = CentroidStream::with_capacity(compiled, bytes.len()).unwrap();
        let mut stats = SearchStats::default();
        for (i, &b) in bytes.iter().enumerate() {
            let (got, s) = profiled.append_profiled(b).unwrap();
            stats = s;
            assert_eq!(got, plain.append(b).unwrap());
            let oracle = dict.tokenize_reference(&bytes[..=i]).unwrap();
            assert_eq!(
                Some(&got),
                oracle.last(),
                "mode={mode:?} endpoint={}",
                i + 1
            );
            assert_eq!(profiled.reconstruct(i + 1).unwrap(), oracle);
            assert_eq!(profiled.prefix(i + 1), plain.prefix(i + 1));
        }
        for endpoint in 0..=bytes.len() {
            assert_eq!(
                profiled.reconstruct(endpoint).unwrap(),
                dict.tokenize_reference(&bytes[..endpoint]).unwrap()
            );
        }
        final_stats.push(stats);
    }
    final_stats
}

/// A random normalized dictionary over four letters, grown one canonical rule
/// at a time.
fn random_dictionary(seed: &mut u64, rules: usize) -> Arc<Dictionary> {
    let mut words: Vec<Vec<u32>> = (0..4).map(|a| vec![a]).collect();
    let mut merges = Vec::new();
    for _ in 0..rules {
        let mut candidates = Vec::new();
        for x in 0..words.len() {
            for y in 0..words.len() {
                let word: Vec<_> = words[x].iter().chain(&words[y]).copied().collect();
                if word.len() <= 24
                    && !words.contains(&word)
                    && literal(word.clone(), &merges, 4) == [x as u32, y as u32]
                {
                    candidates.push((x as u32, y as u32, word));
                }
            }
        }
        if candidates.is_empty() {
            break;
        }
        let chosen = rng(seed) as usize % candidates.len();
        let (x, y, word) = candidates.swap_remove(chosen);
        merges.push((x, y));
        words.push(word);
    }
    Arc::new(Dictionary::new(b"abcd", &merges, 100_000).unwrap())
}

#[test]
fn generated_dictionaries_and_rank_sensitive_regressions() {
    let mut seed = 0x81f917c5;
    for _ in 0..48 {
        let dict = random_dictionary(&mut seed, 20);
        let mut bytes = Vec::new();
        for token in 0..dict.token_count() {
            bytes.extend_from_slice(dict.spelling(token as u32).unwrap());
        }
        bytes.extend((0..256).map(|_| b'a' + (rng(&mut seed) % 4) as u8));
        check(dict, &bytes);
    }
    // Strict self-pair rank comparisons and invalid-centroid upward moves.
    let mut merges = vec![(0, 0)];
    for r in 0..8 {
        let t = 2 + r;
        merges.push((t, t));
    }
    let stats = check(
        Arc::new(Dictionary::new(b"ab", &merges, 10_000).unwrap()),
        &vec![b'a'; 1023],
    );
    assert!(stats.iter().all(|s| s.upward_moves > 0));
    // A long linear successor tree exercises the logarithmic decomposition; the
    // alphabet includes zero and bytes that are not valid UTF-8.
    let mut merges = vec![(0, 1)];
    for r in 0..191 {
        merges.push((0, 256 + r));
    }
    let dict = Arc::new(Dictionary::byte_dictionary(&merges, 30_000).unwrap());
    let mut input = vec![0; 193];
    input.push(1);
    input.extend([255, 128, 0, 1]);
    check(dict, &input);
}

#[test]
fn heap_tokenizer_agrees_with_literal_passes() {
    let mut seed = 0x2545f491;
    for _ in 0..32 {
        let dict = random_dictionary(&mut seed, 24);
        for len in [0, 1, 2, 7, 64, 300] {
            let bytes: Vec<u8> = (0..len)
                .map(|_| b'a' + (rng(&mut seed) % 4) as u8)
                .collect();
            assert_eq!(
                dict.tokenize_heap(&bytes).unwrap(),
                dict.tokenize_reference(&bytes).unwrap()
            );
        }
    }
}

#[test]
fn invalid_dictionaries_are_rejected() {
    assert_eq!(
        Dictionary::new(b"", &[], 10).unwrap_err(),
        Error::InvalidAlphabet
    );
    assert_eq!(
        Dictionary::new(b"aa", &[], 10).unwrap_err(),
        Error::InvalidAlphabet
    );
    assert!(matches!(
        Dictionary::new(b"ab", &[(0, 5)], 10),
        Err(Error::InvalidOperand { rule: 1, token: 5 })
    ));
    // "aa" then "a"+"aa": the second rule's spelling tokenizes as [aa, a].
    assert!(matches!(
        Dictionary::new(b"a", &[(0, 0), (0, 1)], 10),
        Err(Error::NonCanonical { rule: 2, .. })
    ));
    assert!(matches!(
        Dictionary::new(b"ab", &[(0, 1), (2, 1)], 4),
        Err(Error::Limit { .. })
    ));
}

#[test]
fn capacity_errors_limits_reset_and_shared_tables() {
    let dict = Arc::new(Dictionary::new(b"ab", &[(0, 1), (2, 1)], 100).unwrap());
    for mode in MODES {
        assert!(matches!(
            CentroidCompiled::compile(
                dict.clone(),
                &CentroidOptions {
                    search: mode,
                    max_table_bytes: 1
                }
            ),
            Err(Error::Limit { .. })
        ));
        let compiled = compile(dict.clone(), mode);
        assert!(matches!(
            CentroidStream::with_capacity(compiled.clone(), usize::MAX),
            Err(StreamError::Overflow)
        ));
        let mut s = CentroidStream::with_capacity(compiled.clone(), 1).unwrap();
        assert!(s.is_empty());
        assert_eq!(s.reconstruct(0).unwrap(), Vec::<u32>::new());
        assert_eq!(
            s.prefix(0),
            Some(Prefix {
                last: None,
                parent: None
            })
        );
        s.append(b'a').unwrap();
        assert!(matches!(s.append(b'b'), Err(StreamError::Capacity { .. })));
        assert_eq!(s.len(), 1);
        assert_eq!(s.reconstruct(1).unwrap(), vec![0]);
        assert!(matches!(s.reserve(usize::MAX), Err(StreamError::Overflow)));
        s.reserve(3).unwrap();
        assert_eq!(s.capacity(), 4);
        assert_eq!(s.remaining_capacity(), 3);
        assert!(matches!(
            s.extend(b"bx"),
            Err(StreamError::UnknownByte(b'x'))
        ));
        assert_eq!(s.len(), 1);
        assert!(matches!(
            s.extend(b"bbbb"),
            Err(StreamError::Capacity { .. })
        ));
        s.extend(b"bb").unwrap();
        assert_eq!(s.reconstruct(3).unwrap(), vec![3]);
        assert_eq!(s.prefix(4), None);
        assert!(matches!(
            s.reconstruct(4),
            Err(StreamError::InvalidPrefix { .. })
        ));
        s.reset();
        s.extend(b"abb").unwrap();
        assert_eq!(s.reconstruct(3).unwrap(), vec![3]);
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let compiled = compiled.clone();
                std::thread::spawn(move || {
                    let mut s = CentroidStream::with_capacity(compiled, 3).unwrap();
                    s.extend(b"abb").unwrap();
                    s.reconstruct(3).unwrap()
                })
            })
            .collect();
        for w in workers {
            assert_eq!(w.join().unwrap(), vec![3]);
        }
    }
}
