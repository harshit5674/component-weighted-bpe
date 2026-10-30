#!/bin/sh
# Downloads the vocabularies and texts of Section 5 and Appendix E of the paper
# into data/, makes the synthetic inputs, and converts the vocabularies for
# examples/vocab_stats.rs.
#
#   sh scripts/fetch_data.sh
#
# Every file is checked against the SHA-256 hash of the file used in the paper,
# and so is every converted vocabulary. Set CHECK_HASHES=0 to skip the checks.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data
CHECK_HASHES=${CHECK_HASHES:-1}

check() { # file sha256
  [ "$CHECK_HASHES" = 0 ] && return 0
  if ! echo "$2  $1" | shasum -a 256 -c - >/dev/null 2>&1; then
    echo "$1: SHA-256 differs from the file used in the paper" >&2
    exit 1
  fi
}

fetch() { # url file sha256
  [ -f "data/$2" ] || curl -fL --retry 3 -o "data/$2" "$1"
  check "data/$2" "$3"
}

HF=https://huggingface.co
fetch "$HF/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json" gpt2.json \
  8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6
fetch "$HF/EleutherAI/pythia-70m/resolve/a39f36b100fe8a5377810d56c3f4789b9c53ac42/tokenizer.json" pythia.json \
  c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624
fetch "$HF/Qwen/Qwen2.5-0.5B/resolve/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json" qwen2.5.json \
  c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
fetch "$HF/deepseek-ai/DeepSeek-V4-Pro/resolve/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json" deepseek-v4.json \
  8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf
fetch "$HF/allenai/Olmo-3.1-32B-Instruct-SFT/resolve/152782ecc41a86c5cbe3fb6afa68bd90934de48a/tokenizer.json" olmo3.1.json \
  7738a25c46a6043a0e2e605138851baa450269ce9d800af6ca9bf6192e45510c
fetch "$HF/bigscience/bloom/resolve/7f10a99ce7c08f03c7719a586cb2cbda1433ac05/tokenizer.json" bloom.json \
  3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3

fetch https://www.gutenberg.org/files/11/11-0.txt alice.txt \
  a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754
fetch https://www.gutenberg.org/files/1342/1342-0.txt pride-and-prejudice.txt \
  81300b79e8a8d65ac530a97578417d06137e3bbc90622a10a65e5036183d2500
fetch https://raw.githubusercontent.com/python/cpython/v3.13.0/Lib/asyncio/base_events.py python-code.txt \
  06368fda7fba857cedfa5dec24d907b1d04f0a3ac8313e9b059b5191734ea354

python3 scripts/make_inputs.py data
check data/agent-json.txt 407574c4e54dc9e0d1d832fb987731d065e380878ec97e7fdbb8aec6be5bfcf3
check data/repeat-a.txt b44ffb72fcc259676bd80495fef1b44b808ca8f1ffe1b1706a4d7911b0e31f11
check data/repeat-space.txt e94f5162353c4eadcc9e62d4bf7cb5c3a363c698b4be27726ff4006016c0f160
check data/repeat-hyphen.txt dac9b575d063e7a709dc2258c4c750e7a1e1e2c16b846d21d482451cf3d4740a

prepare() { # vocabulary sha256-of-converted-file
  python3 scripts/prepare_vocab.py "data/$1.json" "data/$1.ibpe"
  check "data/$1.ibpe" "$2"
}
prepare gpt2 1f694191e23b52d6b01055d26fb532d2da5c0a09a4dcf20fef5ee1be971ced0c
prepare pythia f2f7f76c3fb284b482335a35b03647b28c34385dd9a5e0df6ad33bfef355bf1a
prepare qwen2.5 f3e6b4f72cc0c2c2c6132d693e60970a5b8a72330ad126513feb8b82d9502481
prepare deepseek-v4 d770432d5b40e40503764d3e5cade9cd5628890524ca0e4fd9c8723c4e71214d
prepare olmo3.1 8210c2cb282c20bd432cdfece082e101aa7efa94cca98777937addd1755c8824
prepare bloom 1af03f343653ca17d6471f281ef94c9782ffe6211ea4ac3eb1b10cd1a7aa249c

echo "All inputs match the paper's. Next, for example:"
echo "  cargo run --release --example vocab_stats -- data/qwen2.5.ibpe data/repeat-space.txt data/alice.txt"
