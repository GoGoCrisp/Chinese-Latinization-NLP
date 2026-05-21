# Eval5: CHID Idiom Cloze Pilot

## Dataset
- Source used: `datasets.load_dataset("clue", "chid")`
- Split used: `validation`
- Passage/context field: `content`
- Blank placeholder format: `#idiomNNNNNN#`
- Candidate idioms field: `candidates`
- Answer/gold field: `answers.text` and `answers.candidate_id`
- Candidate set per blank: shared row-level candidate set; the pilot samples 3 distractors per blank
- Multiple blanks per passage: supported when placeholder-order mapping is unambiguous
- Raw count in selected split: 3218
- Raw count across loaded splits: 91374
- Valid blank-level item count: 23011
- Final 4-choice item count before max-items sampling: 22123
- Evaluated item count: 1000
- Context filtering steps: `[{"context_len_limit": 300, "count": 22123}]`

## Scoring
- Primary scoring mode: `option_text_scoring`
- Candidate labels A/B/C/D are not scored.
- Completion score is mean logprob over candidate idiom tokens conditioned on the shared prompt.
- `add_special_tokens=False`; no EOS token is appended for option-text scoring, matching the prior Eval3 option-text pilot.
- Secondary `candidate_plus_suffix` scoring was not run because this standalone pilot uses the primary option-text definition.

## Results
- Random baseline: 25.00%
- Chinese accuracy: 27.20%
- Pinyin-Diacritic accuracy: 26.10%
- Gap Chinese - Pinyin-Diacritic: 1.10%
- Chinese gap vs baseline: 2.20%
- Pinyin-Diacritic gap vs baseline: 1.10%
- Pinyin-Diacritic candidate collapse count/rate: 0 / 0.00%
- Chinese meaningfully above random: False
- Pinyin-Diacritic meaningfully above random: False

## Interpretation
Both models are near 25%; CHID is too hard or option-text scoring is not effective for these 134M pure pretrained models.

This Eval5 result is a standalone pilot only. It does not modify Eval1, Eval2, Eval3, or Eval4 results, and no models were retrained.
