# What If Chinese Were Latinized?

Trained artifacts for the paper’s Chinese-character and Pinyin experiments.

## Language models

Each download contains one 134M-parameter causal language model, its configuration and its matching 32K tokenizer with EOS. Extract the ZIP and pass the directory to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`.

| Condition | Seed | Final step | Download |
| --- | ---: | ---: | --- |
| Chinese-Origin | 42 | 27176 | [chinese-origin-seed42.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/chinese-origin-seed42.zip) |
| Chinese-Origin | 43 | 27176 | [chinese-origin-seed43.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/chinese-origin-seed43.zip) |
| Chinese-Origin | 44 | 27176 | [chinese-origin-seed44.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/chinese-origin-seed44.zip) |
| Pinyin-token | 42 | 27176 | [pinyin-token-seed42.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/pinyin-token-seed42.zip) |
| Pinyin-token | 43 | 27176 | [pinyin-token-seed43.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/pinyin-token-seed43.zip) |
| Pinyin-token | 44 | 27176 | [pinyin-token-seed44.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/pinyin-token-seed44.zip) |
| Pinyin-source | 42 | 29764 | [pinyin-source-seed42.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/pinyin-source-seed42.zip) |
| Pinyin-source | 43 | 29764 | [pinyin-source-seed43.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/pinyin-source-seed43.zip) |
| Pinyin-source | 44 | 29764 | [pinyin-source-seed44.zip](https://github.com/GoGoCrisp/Chinese-Latinization-NLP/releases/download/paper-models-v1/pinyin-source-seed44.zip) |

## Tokenizers

The 16 final SuperBPE tokenizers cover four representations at 8K, 16K, 32K and 64K, with transition at K=2048. Load any `superbpe/*/tokenizer.json` with `tokenizers.Tokenizer.from_file`. The model archives include the EOS-equipped tokenizers used for language modeling.

For the 64K character- and syllable-initialized comparison, use the included inference adapter with `tokenizers==0.22.2`.

```python
from pathlib import Path
from atomic_bpe import AtomicBPE
tokenizer = AtomicBPE.load(Path("atomic/pinyin_diacritic_64000"))
ids = tokenizer.encode("zhōng guó")
assert tokenizer.decode(ids) == "zhōng guó"
```
