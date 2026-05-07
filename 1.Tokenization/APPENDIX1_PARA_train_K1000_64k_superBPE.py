import shutil
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "superTokenizers_BPE_K1000_64k_subset100k"
STAGE1_SOURCE_ROOTS = [
    BASE_DIR / "superTokenizers_BPE_2048_subset100k",
    BASE_DIR / "superTokenizers_BPE",
]

SUPERBPE_VENV_PYTHON = BASE_DIR / ".." / "superbpe" / "superbpe_venv" / "bin" / "python"
TRAIN_TOKENIZER_SCRIPT = (BASE_DIR / ".." / "superbpe" / "train_tokenizer.py").resolve()

FIXED_K = 1000
VOCAB_SIZE = 64000
STAGE2_REGEX = r"[^\p{L}\p{N}\s]+|[\r\n]+"

CORPORA = [
    "chinese_origin",
    "pinyin_toned",
    "pinyin_toneless",
    "pinyin_diacritic",
]


def find_stage1_dir(name: str) -> Path:
    dirname = f"{name}_subset100k_stage1_{VOCAB_SIZE}"
    for root in STAGE1_SOURCE_ROOTS:
        candidate = root / dirname
        if (candidate / "merges.txt").exists() and (candidate / "meta.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find reusable stage1 tokenizer for {dirname}")


def prepare_initial_merges(stage1_dir: Path, stage2_dir: Path) -> None:
    stage2_dir.mkdir(parents=True, exist_ok=True)

    with open(stage1_dir / "merges.txt", "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    # merges.txt includes a #version header. Keep header + K actual merges.
    with open(stage2_dir / "merges.txt", "w", encoding="utf-8") as f_out:
        f_out.writelines(lines[: FIXED_K + 1])

    shutil.copy(stage1_dir / "meta.json", stage2_dir / "meta.json")


def train_one(name: str) -> None:
    stage1_dir = find_stage1_dir(name)
    stage2_dir = OUTPUT_ROOT / f"{name}_subset100k_superbpe_{VOCAB_SIZE}"

    if (stage2_dir / "tokenizer.json").exists():
        print(f"Skip existing tokenizer: {stage2_dir}")
        return

    print(f"\nTraining {name} {VOCAB_SIZE} SuperBPE with fixed K={FIXED_K}")
    prepare_initial_merges(stage1_dir, stage2_dir)

    cmd = [
        str(SUPERBPE_VENV_PYTHON),
        str(TRAIN_TOKENIZER_SCRIPT),
        "--output_dir",
        str(stage2_dir),
        "--vocab_size",
        str(VOCAB_SIZE),
        "--regex_string",
        STAGE2_REGEX,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in CORPORA:
        train_one(name)


if __name__ == "__main__":
    main()
