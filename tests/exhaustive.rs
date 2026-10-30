use component_weighted_bpe::*;
use std::sync::Arc;
mod support;
use support::{enumerate, literal};

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

/// Every normalized dictionary over {a, b} with at most three rules and token
/// spellings of at most eight bytes, on every input of seven bytes.
#[test]
fn both_searches_match_literal_bpe_on_all_small_dictionaries() {
    let (mut dictionaries, mut comparisons) = (0, 0);
    enumerate(Vec::new(), vec![vec![0], vec![1]], &mut |merges| {
        dictionaries += 1;
        let dict = Arc::new(Dictionary::new(b"ab", merges, 1000).unwrap());
        let mut streams: Vec<_> = MODES
            .into_iter()
            .map(|mode| CentroidStream::with_capacity(compile(dict.clone(), mode), 7).unwrap())
            .collect();
        for input in 0..128 {
            for stream in &mut streams {
                stream.reset();
            }
            let atoms: Vec<_> = (0..7)
                .rev()
                .map(|bit| ((input >> bit) & 1) as u32)
                .collect();
            for i in 0..7 {
                let expected = literal(atoms[..=i].to_vec(), merges, 2);
                let byte = b'a' + atoms[i] as u8;
                for stream in &mut streams {
                    let (got, stats) = stream.append_profiled(byte).unwrap();
                    assert_eq!(
                        Some(&got),
                        expected.last(),
                        "rules={merges:?}, input={atoms:?}, endpoint={}",
                        i + 1
                    );
                    assert_eq!(stream.reconstruct(i + 1).unwrap(), expected);
                    assert!(stats.terminal_misses <= 1);
                    assert!(
                        stats.centroid_visits <= stream.compiled().report().max_centroid_levels
                    );
                    comparisons += 1;
                }
                assert_eq!(streams[0].prefix(i + 1), streams[1].prefix(i + 1));
            }
            for endpoint in 0..=7 {
                for stream in &streams {
                    assert_eq!(
                        stream.reconstruct(endpoint).unwrap(),
                        literal(atoms[..endpoint].to_vec(), merges, 2)
                    );
                }
            }
        }
    });
    assert_eq!((dictionaries, comparisons), (399, 715_008));
}
