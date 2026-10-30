// Each test binary uses only some of these helpers.
#![allow(dead_code)]
pub fn literal(mut seq: Vec<u32>, rules: &[(u32, u32)], atoms: usize) -> Vec<u32> {
    for (r, &(x, y)) in rules.iter().enumerate() {
        let mut result = Vec::new();
        let mut i = 0;
        while i < seq.len() {
            if i + 1 < seq.len() && seq[i] == x && seq[i + 1] == y {
                result.push((atoms + r) as u32);
                i += 2;
            } else {
                result.push(seq[i]);
                i += 1;
            }
        }
        seq = result;
    }
    seq
}

pub fn enumerate(
    rules: Vec<(u32, u32)>,
    words: Vec<Vec<u32>>,
    visit: &mut impl FnMut(&[(u32, u32)]),
) {
    visit(&rules);
    if rules.len() == 3 {
        return;
    }
    for x in 0..words.len() {
        for y in 0..words.len() {
            let mut word = words[x].clone();
            word.extend_from_slice(&words[y]);
            if word.len() > 8 || words.contains(&word) {
                continue;
            }
            if literal(word.clone(), &rules, 2) != [x as u32, y as u32] {
                continue;
            }
            let mut next_rules = rules.clone();
            next_rules.push((x as u32, y as u32));
            let mut next_words = words.clone();
            next_words.push(word);
            enumerate(next_rules, next_words, visit);
        }
    }
}

pub fn rng(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}
