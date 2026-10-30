#!/bin/sh
# Downloads the vocabularies and texts used in Section 5 of the paper into
# data/ and prepares the IBPE files for examples/vocab_stats.rs.
#
#   sh scripts/fetch_data.sh
#
# The paper's exact snapshots are not distributed. This script downloads the
# current files and records their SHA-256 hashes in data/SHA256SUMS, and the
# TheAlgorithms/Rust commit in data/rust-algorithms.commit.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {
  [ -f "$2" ] || curl -fL --retry 3 -o "$2" "$1"
}

fetch https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json data/gpt2.json
fetch https://huggingface.co/FacebookAI/roberta-base/resolve/main/tokenizer.json data/roberta-base.json
fetch https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json data/gpt-neox-20b.json
fetch https://www.gutenberg.org/cache/epub/2600/pg2600.txt data/war-and-peace.txt
[ -d data/rust-algorithms ] || git clone --depth 1 https://github.com/TheAlgorithms/Rust data/rust-algorithms
git -C data/rust-algorithms rev-parse HEAD > data/rust-algorithms.commit

# One million bytes of English prose and of Rust code.
head -c 1000000 data/war-and-peace.txt > data/english.txt
find data/rust-algorithms -name '*.rs' | LC_ALL=C sort | xargs cat | head -c 1000000 > data/code.txt

for vocab in gpt2 roberta-base gpt-neox-20b; do
  python3 scripts/prepare_vocab.py "data/$vocab.json" "data/$vocab.ibpe"
done

(cd data && shasum -a 256 gpt2.json roberta-base.json gpt-neox-20b.json \
  war-and-peace.txt english.txt code.txt gpt2.ibpe roberta-base.ibpe gpt-neox-20b.ibpe > SHA256SUMS)
echo "Done. Next: cargo run --release --example vocab_stats -- data/gpt2.ibpe data/english.txt data/code.txt"
