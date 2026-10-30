//! Tokenize a short input byte by byte with both branch searches.
//!
//!     cargo run --example quickstart
use component_weighted_bpe::*;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Alphabet "ab". Rule 0: a+a -> aa (token 2). Rule 1: aa+aa -> aaaa (token 3).
    // Rule 2: aaaa+aaaa -> aaaaaaaa (token 4).
    let dictionary = Arc::new(Dictionary::new(b"ab", &[(0, 0), (2, 2), (3, 3)], 100)?);
    let input = b"aaaaaaaaaaaab";
    for search in [
        CentroidSearch::CountBalanced,
        CentroidSearch::ComponentWeighted,
    ] {
        let options = CentroidOptions {
            search,
            ..Default::default()
        };
        let compiled = Arc::new(CentroidCompiled::compile(dictionary.clone(), &options)?);
        let mut stream = CentroidStream::with_capacity(compiled, input.len())?;
        println!("{search:?}");
        for (i, &byte) in input.iter().enumerate() {
            let (last, stats) = stream.append_profiled(byte)?;
            println!(
                "  prefix {:2}: last token {last}, centroids visited {}, interval probes {}",
                i + 1,
                stats.centroid_visits,
                stats.interval_probes
            );
        }
        let tokens = stream.reconstruct(input.len())?;
        assert_eq!(tokens, dictionary.tokenize_reference(input)?);
        println!("  tokens: {tokens:?}");
    }
    Ok(())
}
