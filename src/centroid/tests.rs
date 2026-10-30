use super::compiler::weighted_links;
use super::search::{count_balanced, weighted};
use super::*;

fn rng(x: &mut u64) -> usize {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x as usize
}

#[test]
fn interval_maps_cover_boundaries_gaps_and_weight_ratio_bounds() {
    let mut seed = 0x671345ab;
    for n in 0..=64 {
        for shape in 0..4 {
            let mut intervals = Vec::new();
            let mut end = rng(&mut seed) % 5;
            for i in 0..n {
                let lo = end + rng(&mut seed) % 4;
                end = lo + 1 + rng(&mut seed) % 4;
                let weight = match shape {
                    0 => 1,
                    1 if i == 0 => 100_000,
                    2 if i == n - 1 => 100_000,
                    _ => 1 + rng(&mut seed) % 100,
                };
                intervals.push(Interval {
                    lo,
                    hi: end,
                    weight,
                    target: i,
                });
            }
            let mut links = vec![
                Links {
                    left: NONE,
                    right: NONE
                };
                n
            ];
            let root = weighted_links(&intervals, &mut links, 0).unwrap();
            let total: usize = intervals.iter().map(|x| x.weight).sum();
            for x in 0..=end + 3 {
                let expected = intervals
                    .iter()
                    .position(|i| i.lo <= x && x < i.hi)
                    .unwrap_or(NONE);
                let mut stats = SearchStats::default();
                assert_eq!(
                    weighted::<true>(&intervals, &links, root, x, &mut stats),
                    expected
                );
                assert_eq!(
                    count_balanced::<true>(&intervals, 0..n, x, &mut SearchStats::default()),
                    expected
                );
                if n != 0 {
                    let w = if expected == NONE {
                        1
                    } else {
                        intervals[expected].weight
                    };
                    let mut doubled = w;
                    let mut ceil_log = 0;
                    while doubled < total {
                        doubled *= 2;
                        ceil_log += 1;
                    }
                    assert!(stats.interval_probes <= 1 + ceil_log);
                } else {
                    assert_eq!(stats.interval_probes, 0);
                }
            }
            fn check(
                v: usize,
                links: &[Links],
                intervals: &[Interval],
                seen: &mut [bool],
            ) -> usize {
                if v == NONE {
                    return 0;
                }
                assert!(!seen[v]);
                seen[v] = true;
                if links[v].left != NONE {
                    assert!(links[v].left < v);
                }
                if links[v].right != NONE {
                    assert!(links[v].right > v);
                }
                let l = check(links[v].left, links, intervals, seen);
                let r = check(links[v].right, links, intervals, seen);
                let total = l + r + intervals[v].weight;
                assert!(l <= total / 2 && r <= total / 2);
                total
            }
            let mut seen = vec![false; n];
            assert_eq!(check(root, &links, &intervals, &mut seen), total);
            assert!(seen.into_iter().all(|x| x));
        }
    }
}

#[test]
fn centroid_tables_have_true_component_weights_and_no_omitted_suffixes() {
    // Includes suffix candidates that are not on the matched token's own
    // generating-successor chain, plus overlapping self-pairs.
    let dict = Arc::new(Dictionary::new(b"abc", &[(0, 0), (0, 1), (3, 4), (2, 4)], 1000).unwrap());
    for search in [
        CentroidSearch::CountBalanced,
        CentroidSearch::ComponentWeighted,
    ] {
        let compiled = CentroidCompiled::compile(
            dict.clone(),
            &CentroidOptions {
                search,
                ..Default::default()
            },
        )
        .unwrap();
        for t in 0..dict.token_count() {
            let word = dict.spelling(t as u32).unwrap();
            let expected = (0..dict.token_count())
                .filter(|&v| word.ends_with(dict.spelling(v as u32).unwrap()))
                .count();
            let mut pending = vec![compiled.roots[t]];
            let mut seen = std::collections::BTreeSet::new();
            while let Some(v) = pending.pop() {
                let node = &compiled.centroids[v];
                assert!(seen.insert(node.token));
                assert!(word.ends_with(dict.spelling(node.token).unwrap()));
                let mut sum = 1;
                if node.up != NONE {
                    let size = compiled.centroids[node.up].size;
                    assert!(size <= node.size / 2);
                    sum += size;
                    pending.push(node.up);
                }
                for interval in &compiled.intervals[node.down.clone()] {
                    assert_eq!(interval.weight, compiled.centroids[interval.target].size);
                    assert!(interval.weight <= node.size / 2);
                    sum += interval.weight;
                    pending.push(interval.target);
                }
                assert_eq!(sum, node.size);
            }
            assert_eq!(seen.len(), expected);
        }
        assert_eq!(
            compiled.links.is_empty(),
            search == CentroidSearch::CountBalanced
        );
    }
}
