import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "superTokenizers_BPE_K1000_64k_subset100k"
OUTPUT_DIR = BASE_DIR / "decoded_superTokenizers_K1000_64k_subset100k"


def decode_tokenizer(tokenizer_path: Path, output_path: Path) -> None:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.decoder = ByteLevel()

    decoded_vocab = {}
    for _raw_token, token_id in tokenizer.get_vocab().items():
        decoded_vocab[tokenizer.decode([token_id])] = token_id

    sorted_vocab = dict(sorted(decoded_vocab.items(), key=lambda item: item[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)


def main() -> None:
    count = 0
    for tokenizer_path in sorted(INPUT_DIR.glob("*_superbpe_64000/tokenizer.json")):
        tokenizer_name = tokenizer_path.parent.name
        output_path = OUTPUT_DIR / f"{tokenizer_name}_decoded.json"
        decode_tokenizer(tokenizer_path, output_path)
        count += 1
        print(f"Decoded {tokenizer_name}")

    print(f"Done. Decoded {count} tokenizers into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
