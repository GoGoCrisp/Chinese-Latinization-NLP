#!/usr/bin/env python3
"""Build Eval 4 Chinese BLiMP-style minimal-pair data.

The builder first searches for local Chinese BLiMP/Zho-BLMP style data. If no
valid local dataset is found, it creates a fixed-seed template dataset covering
general linguistic contrasts that should not depend on factual knowledge.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pypinyin import Style, lazy_pinyin


DEFAULT_OUTPUT_DIR = Path("eval_data/eval4_chinese_blimp_style")
DEFAULT_JSONL = "eval4_chinese_blimp_style.jsonl"
DEFAULT_REVIEW = "eval4_chinese_blimp_style_review.csv"
DEFAULT_TARGET_ITEMS = 2000
DEFAULT_SEED = 42
DEFAULT_HF_DATASET = "chinese-babylm-org/zhoblimp"

FIELDS = [
    "id",
    "phenomenon",
    "subtype_if_any",
    "good_sentence_zh",
    "bad_sentence_zh",
    "good_sentence_diacritic",
    "bad_sentence_diacritic",
    "data_source",
    "generation_method",
    "quality_flags",
]

SEARCH_TERMS = [
    "zho_blimp",
    "zhoblimp",
    "chinese_blimp",
    "blimp_zh",
    "minimal_pairs",
    "minimal-pairs",
    "linguistic_minimal_pairs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Eval 4 Chinese BLiMP-style minimal-pair data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-items", type=int, default=DEFAULT_TARGET_ITEMS)
    parser.add_argument("--all-items", action="store_true", help="Use all valid ZhoBLiMP/local items instead of sampling.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-custom", action="store_true")
    parser.add_argument(
        "--prefer-zhoblimp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/load ZhoBLiMP from Hugging Face when no valid local dataset is found.",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--print-examples", type=int, default=12)
    return parser.parse_args()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def to_diacritic(text: str) -> str:
    parts = lazy_pinyin(
        str(text),
        style=Style.TONE,
        neutral_tone_with_five=False,
        errors=lambda chunk: list(chunk),
    )
    return re.sub(r"\s+", " ", " ".join(part.strip() for part in parts if part.strip())).strip()


def quality_flags(good_zh: str, bad_zh: str, good_py: str, bad_py: str) -> list[str]:
    flags: list[str] = []
    if good_zh == bad_zh:
        flags.append("identical_zh")
    if good_py == bad_py:
        flags.append("identical_diacritic")
    good_tokens = good_py.split()
    bad_tokens = bad_py.split()
    if good_tokens and bad_tokens:
        overlap = len(set(good_tokens) & set(bad_tokens)) / max(len(set(good_tokens) | set(bad_tokens)), 1)
        if overlap >= 0.95 and good_tokens != bad_tokens:
            flags.append("near_identical_diacritic_bag")
    if abs(len(good_zh) - len(bad_zh)) > 3:
        flags.append("length_delta_gt_3_chars")
    return flags


def make_item(
    index: int,
    phenomenon: str,
    subtype: str,
    good_zh: str,
    bad_zh: str,
    data_source: str,
    generation_method: str,
) -> dict[str, Any]:
    good_py = to_diacritic(good_zh)
    bad_py = to_diacritic(bad_zh)
    return {
        "id": f"eval4_{index:05d}",
        "phenomenon": phenomenon,
        "subtype_if_any": subtype,
        "good_sentence_zh": good_zh,
        "bad_sentence_zh": bad_zh,
        "good_sentence_diacritic": good_py,
        "bad_sentence_diacritic": bad_py,
        "data_source": data_source,
        "generation_method": generation_method,
        "quality_flags": quality_flags(good_zh, bad_zh, good_py, bad_py),
    }


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "items", "examples", "records"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_candidate_row(raw: dict[str, Any], source_path: Path, row_index: int) -> dict[str, Any] | None:
    good = raw.get("good_sentence") or raw.get("sentence_good") or raw.get("good_sentence_zh")
    bad = raw.get("bad_sentence") or raw.get("sentence_bad") or raw.get("bad_sentence_zh")
    phenomenon = raw.get("phenomenon") or raw.get("category") or raw.get("linguistic_phenomenon")
    if not good or not bad or not phenomenon:
        return None
    subtype = raw.get("subtype_if_any") or raw.get("subtype") or raw.get("field") or ""
    item = make_item(
        index=row_index,
        phenomenon=str(phenomenon).strip(),
        subtype=str(subtype).strip(),
        good_zh=str(good).strip(),
        bad_zh=str(bad).strip(),
        data_source=str(source_path),
        generation_method="normalized_existing_local_dataset",
    )
    return item


def normalize_zhoblimp_row(raw: dict[str, Any], row_index: int, source_name: str) -> dict[str, Any] | None:
    good = raw.get("sentence_good") or raw.get("good_sentence") or raw.get("good_sentence_zh")
    bad = raw.get("sentence_bad") or raw.get("bad_sentence") or raw.get("bad_sentence_zh")
    phenomenon = raw.get("phenomenon") or raw.get("field") or raw.get("category")
    if not good or not bad or not phenomenon:
        return None
    uid = raw.get("UID") or raw.get("uid") or raw.get("paradigm") or raw.get("subtype") or ""
    pair_id = raw.get("pairID") or raw.get("pair_id") or raw.get("id") or row_index
    item = make_item(
        index=row_index,
        phenomenon=str(phenomenon).strip(),
        subtype=str(uid).strip(),
        good_zh=str(good).strip(),
        bad_zh=str(bad).strip(),
        data_source=f"hf://{source_name}#pairID={pair_id}",
        generation_method="downloaded_zhoblimp",
    )
    return item


def stratified_sample_items(
    items: list[dict[str, Any]],
    target_items: int,
    seed: int,
    all_items: bool = False,
) -> list[dict[str, Any]]:
    if target_items <= 0 and not all_items:
        raise ValueError(f"--target-items must be positive unless --all-items is set; got {target_items}")
    if all_items or len(items) <= target_items:
        sampled = list(items)
    else:
        rng = random.Random(seed)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[item["phenomenon"]].append(item)
        for rows in groups.values():
            rng.shuffle(rows)

        phenomena = sorted(groups)
        base = target_items // len(phenomena)
        remainder = target_items % len(phenomena)
        sampled = []
        leftovers: list[dict[str, Any]] = []
        for idx, phenomenon in enumerate(phenomena):
            quota = base + (1 if idx < remainder else 0)
            rows = groups[phenomenon]
            take = min(quota, len(rows))
            sampled.extend(rows[:take])
            leftovers.extend(rows[take:])
        if len(sampled) < target_items:
            rng.shuffle(leftovers)
            sampled.extend(leftovers[: target_items - len(sampled)])
        rng.shuffle(sampled)

    for index, item in enumerate(sampled):
        item["id"] = f"eval4_{index:05d}"
        if item["generation_method"] == "downloaded_zhoblimp":
            item["generation_method"] = (
                "downloaded_zhoblimp_all_items"
                if all_items
                else f"downloaded_zhoblimp_stratified_sample_seed{seed}"
            )
    return sampled


def load_zhoblimp_hf(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"ZhoBLiMP unavailable: cannot import datasets ({exc})")
        return [], None

    split_candidates = [args.hf_split] if args.hf_split else ["train", "test", "validation", "dev"]
    loaded = None
    used_split = None
    last_error: Exception | None = None
    for split in split_candidates:
        try:
            loaded = load_dataset(args.hf_dataset, split=split)
            used_split = split
            break
        except Exception as exc:
            last_error = exc
            continue
    if loaded is None:
        try:
            dataset_dict = load_dataset(args.hf_dataset)
            used_split = sorted(dataset_dict.keys())[0]
            loaded = dataset_dict[used_split]
        except Exception as exc:
            last_error = exc
    if loaded is None:
        print(f"ZhoBLiMP unavailable: failed to load {args.hf_dataset} ({last_error})")
        return [], None

    raw_features = list(getattr(loaded, "features", {}).keys())
    normalized = []
    for idx, raw in enumerate(loaded):
        item = normalize_zhoblimp_row(dict(raw), idx, args.hf_dataset)
        if item is not None:
            normalized.append(item)
    if not normalized:
        print(f"ZhoBLiMP loaded from {args.hf_dataset} but no valid minimal-pair rows were found")
        return [], None

    sampled = stratified_sample_items(normalized, args.target_items, args.seed, args.all_items)
    meta = {
        "source": "zhoblimp_huggingface",
        "dataset": args.hf_dataset,
        "split": used_split,
        "n_loaded_items": len(normalized),
        "n_items": len(sampled),
        "all_items": bool(args.all_items),
        "target_items": args.target_items,
        "raw_fields": raw_features,
        "fields": FIELDS,
        "categories": dict(Counter(item["phenomenon"] for item in sampled)),
        "uids": len(set(item["subtype_if_any"] for item in normalized if item["subtype_if_any"])),
    }
    return sampled, meta


def discover_existing_dataset(root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    search_roots = [root]
    search_roots.extend(project_path(root, value) for value in args.search_root)
    seen: set[Path] = set()
    candidates: list[Path] = []
    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jsonl", ".json", ".csv", ".tsv"}:
                continue
            lower = str(path).lower()
            if any(term in lower for term in SEARCH_TERMS):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(path)

    valid_sets: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in sorted(candidates):
        try:
            if path.suffix.lower() in {".jsonl", ".json"}:
                raw_rows = read_json_or_jsonl(path)
            else:
                raw_rows = read_csv_rows(path)
        except Exception as exc:
            print(f"candidate unreadable: {path} ({exc})")
            continue
        normalized = [
            item
            for idx, row in enumerate(raw_rows)
            if (item := normalize_candidate_row(row, path, idx)) is not None
        ]
        if normalized:
            valid_sets.append((path, normalized))

    if not valid_sets:
        return [], None
    valid_sets.sort(key=lambda pair: len(pair[1]), reverse=True)
    path, items = valid_sets[0]
    for index, item in enumerate(items):
        item["id"] = f"eval4_existing_{index:05d}"
    meta = {
        "source": "existing_local_dataset",
        "path": str(path),
        "n_items": len(items),
        "fields": FIELDS,
        "categories": dict(Counter(item["phenomenon"] for item in items)),
        "searched_candidate_files": [str(path) for path in candidates],
    }
    return items, meta


def classifier_pairs() -> list[tuple[str, str, str]]:
    nouns = [
        ("人", "个", "本"), ("学生", "名", "本"), ("老师", "位", "张"), ("医生", "位", "条"),
        ("书", "本", "个"), ("小说", "本", "只"), ("字典", "本", "位"), ("杂志", "本", "条"),
        ("信", "封", "个"), ("照片", "张", "位"), ("桌子", "张", "条"), ("椅子", "把", "本"),
        ("车", "辆", "张"), ("自行车", "辆", "本"), ("狗", "只", "本"), ("猫", "只", "张"),
        ("鱼", "条", "本"), ("河", "条", "本"), ("路", "条", "个"), ("裤子", "条", "本"),
        ("衣服", "件", "条"), ("衬衫", "件", "辆"), ("花", "朵", "本"), ("树", "棵", "条"),
        ("房子", "间", "条"), ("办公室", "间", "本"), ("电影", "部", "张"), ("电脑", "台", "匹"),
        ("机器", "台", "条"), ("马", "匹", "张"), ("船", "艘", "本"), ("山", "座", "条"),
    ]
    nums = ["一", "两", "三", "四", "五", "几"]
    prefixes = ["我看见了", "桌上有", "他买了", "她需要", "我们借了", "门口停着", "那里来了", "箱子里有"]
    suffixes = ["。", "，很快就用上了。", "，大家都注意到了。"]
    pairs = []
    for noun, good_clf, bad_clf in nouns:
        for num in nums:
            for prefix in prefixes:
                suffix = suffixes[(len(noun) + len(prefix) + len(num)) % len(suffixes)]
                pairs.append((f"{prefix}{num}{good_clf}{noun}{suffix}", f"{prefix}{num}{bad_clf}{noun}{suffix}", noun))
    return pairs


def word_order_pairs() -> list[tuple[str, str, str]]:
    subjects = ["他", "她", "老师", "学生", "司机", "妈妈", "同事", "朋友", "孩子", "记者"]
    times = ["昨天", "今天上午", "明天", "周末", "晚上", "刚才", "去年", "下课后"]
    verbs_objects = [
        ("去了", "北京"), ("回到", "家里"), ("参观了", "学校"), ("整理了", "房间"),
        ("写完了", "报告"), ("看见了", "朋友"), ("离开了", "办公室"), ("打开了", "窗户"),
        ("买了", "水果"), ("完成了", "作业"),
    ]
    pairs = []
    for subject in subjects:
        for time in times:
            for verb, obj in verbs_objects:
                pairs.append((f"{subject}{time}{verb}{obj}。", f"{subject}{verb}{time}{obj}。", time))
    return pairs


def negation_pairs() -> list[tuple[str, str, str]]:
    subjects = ["他", "她", "老师", "学生", "医生", "经理", "孩子", "朋友", "司机", "同事"]
    verbs_objects = [
        ("去", "学校"), ("看", "电影"), ("吃", "早饭"), ("写", "作业"), ("参加", "会议"),
        ("打开", "门"), ("收到", "通知"), ("完成", "任务"), ("买", "票"), ("整理", "资料"),
    ]
    adverbs = ["今天", "昨天", "上午", "晚上", "刚才"]
    pairs = []
    for subject in subjects:
        for adv in adverbs:
            for verb, obj in verbs_objects:
                pairs.append((f"{subject}{adv}没有{verb}{obj}。", f"{subject}{adv}{verb}没有{obj}。", "没有"))
                pairs.append((f"{subject}{adv}不想{verb}{obj}。", f"{subject}{adv}{verb}不想{obj}。", "不想"))
    return pairs


def aspect_pairs() -> list[tuple[str, str, str]]:
    subjects = ["他", "她", "老师", "学生", "妈妈", "朋友", "同事", "孩子", "记者", "司机"]
    verb_objects = [
        ("吃", "饭"), ("喝", "水"), ("写", "信"), ("看", "书"), ("买", "菜"),
        ("修", "车"), ("关", "门"), ("洗", "衣服"), ("读", "文章"), ("做", "作业"),
    ]
    times = ["刚才", "昨天", "早上", "中午", "晚上"]
    pairs = []
    for subject in subjects:
        for time in times:
            for verb, obj in verb_objects:
                pairs.append((f"{subject}{time}{verb}了{obj}。", f"{subject}{time}了{verb}{obj}。", "了"))
                pairs.append((f"{subject}{time}{verb}过{obj}。", f"{subject}{time}过{verb}{obj}。", "过"))
    return pairs


def de_particle_pairs() -> list[tuple[str, str, str]]:
    adj_nouns = [
        ("美丽", "花"), ("安静", "房间"), ("干净", "衣服"), ("温暖", "阳光"), ("新鲜", "水果"),
        ("清楚", "答案"), ("重要", "消息"), ("漂亮", "照片"), ("认真", "态度"), ("简单", "办法"),
    ]
    adv_verbs = [
        ("认真", "学习"), ("安静", "等待"), ("慢慢", "走路"), ("仔细", "检查"), ("努力", "工作"),
        ("高兴", "唱歌"), ("清楚", "说明"), ("耐心", "解释"), ("快速", "移动"), ("轻轻", "关门"),
    ]
    verb_comps = [
        ("跑", "很快"), ("写", "很清楚"), ("说", "很流利"), ("笑", "很开心"), ("睡", "很香"),
        ("打扫", "很干净"), ("解释", "很详细"), ("画", "很好"), ("走", "很慢"), ("唱", "很好听"),
    ]
    prefixes = ["我喜欢", "大家看见了", "这里有", "他注意到", "她记得"]
    pairs = []
    for adj, noun in adj_nouns:
        for prefix in prefixes:
            pairs.append((f"{prefix}{adj}的{noun}。", f"{prefix}{adj}地{noun}。", "的_vs_地"))
    for adv, verb in adv_verbs:
        for subject in ["他", "她", "学生", "老师", "大家"]:
            pairs.append((f"{subject}{adv}地{verb}。", f"{subject}{adv}得{verb}。", "地_vs_得"))
    for verb, comp in verb_comps:
        for subject in ["他", "她", "孩子", "朋友", "同事"]:
            pairs.append((f"{subject}{verb}得{comp}。", f"{subject}{verb}的{comp}。", "得_vs_的"))
    return pairs


def selectional_pairs() -> list[tuple[str, str, str]]:
    rows = [
        ("喝", "水", "石头"), ("喝", "茶", "桌子"), ("喝", "牛奶", "空气"), ("吃", "饭", "空气"),
        ("吃", "面包", "声音"), ("吃", "苹果", "阳光"), ("阅读", "文章", "桌子"), ("阅读", "小说", "杯子"),
        ("阅读", "报告", "椅子"), ("驾驶", "汽车", "铅笔"), ("驾驶", "公交车", "纸张"), ("修理", "机器", "天气"),
        ("修理", "电脑", "月光"), ("穿", "衣服", "问题"), ("穿", "鞋子", "消息"), ("听见", "声音", "石头"),
        ("听见", "歌声", "面包"), ("闻到", "香味", "数字"), ("种植", "树苗", "玻璃"), ("浇", "花", "句子"),
    ]
    subjects = ["他", "她", "老师", "学生", "妈妈", "朋友", "工人", "孩子", "同事", "大家"]
    adverbs = ["正在", "喜欢", "准备", "经常", "刚刚"]
    pairs = []
    for verb, good_obj, bad_obj in rows:
        for subject in subjects:
            for adv in adverbs:
                pairs.append((f"{subject}{adv}{verb}{good_obj}。", f"{subject}{adv}{verb}{bad_obj}。", verb))
    return pairs


def collocation_pairs() -> list[tuple[str, str, str]]:
    rows = [
        ("做", "吃", "决定"), ("提出", "穿", "问题"), ("解决", "喝", "问题"), ("采取", "吃", "措施"),
        ("获得", "穿", "成功"), ("积累", "喝", "经验"), ("提供", "睡", "帮助"), ("制定", "喝", "计划"),
        ("发表", "穿", "意见"), ("承担", "吃", "责任"), ("保持", "喝", "联系"), ("引起", "穿", "注意"),
        ("开展", "吃", "活动"), ("改善", "喝", "条件"), ("增强", "穿", "信心"), ("降低", "睡", "风险"),
        ("提高", "吃", "效率"), ("建立", "喝", "关系"), ("完成", "穿", "任务"), ("接受", "睡", "建议"),
    ]
    subjects = ["他", "她", "老师", "学生", "公司", "团队", "部门", "大家", "我们", "同事"]
    modifiers = ["已经", "正在", "准备", "需要", "努力"]
    pairs = []
    for good_v, bad_v, obj in rows:
        for subject in subjects:
            for modifier in modifiers:
                pairs.append((f"{subject}{modifier}{good_v}{obj}。", f"{subject}{modifier}{bad_v}{obj}。", obj))
    return pairs


def function_word_pairs() -> list[tuple[str, str, str]]:
    rows = [
        ("我", "喜欢", "这本书"), ("他", "认识", "那个老师"), ("她", "看见", "一只猫"), ("学生", "完成", "作业"),
        ("老师", "打开", "窗户"), ("朋友", "买了", "水果"), ("孩子", "拿着", "玩具"), ("妈妈", "整理", "房间"),
        ("司机", "开着", "车"), ("同事", "收到", "邮件"), ("我们", "讨论", "计划"), ("大家", "听见", "声音"),
        ("医生", "检查", "病人"), ("记者", "采访", "学生"), ("经理", "安排", "会议"), ("工人", "修理", "机器"),
        ("老人", "读着", "报纸"), ("女孩", "画了", "图画"), ("男孩", "踢着", "足球"), ("邻居", "借了", "雨伞"),
    ]
    times = ["今天", "昨天", "上午", "刚才", "晚上"]
    pairs = []
    for subject, verb, obj in rows:
        for time in times:
            pairs.append((f"{subject}{time}{verb}{obj}。", f"{subject}{time}{verb}的{obj}。", "spurious_的"))
            pairs.append((f"{subject}{time}把{obj}{verb}。", f"{subject}{time}{obj}{verb}。", "missing_把"))
    return pairs


PHENOMENON_GENERATORS = {
    "classifier_noun_mismatch": classifier_pairs,
    "word_order": word_order_pairs,
    "negation_position": negation_pairs,
    "aspect_marker_position": aspect_pairs,
    "de_particle": de_particle_pairs,
    "selectional_preference": selectional_pairs,
    "local_collocation": collocation_pairs,
    "function_word_insertion_or_deletion": function_word_pairs,
}


def build_custom_dataset(target_items: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    phenomena = list(PHENOMENON_GENERATORS)
    base_per_phenomenon = target_items // len(phenomena)
    remainder = target_items % len(phenomena)
    desired = {
        phenomenon: base_per_phenomenon + (1 if idx < remainder else 0)
        for idx, phenomenon in enumerate(phenomena)
    }
    items: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    index = 0
    for phenomenon in phenomena:
        pairs = PHENOMENON_GENERATORS[phenomenon]()
        rng.shuffle(pairs)
        if len(pairs) < desired[phenomenon]:
            raise RuntimeError(f"Only {len(pairs)} template pairs available for {phenomenon}; need {desired[phenomenon]}")
        selected: list[tuple[str, str, str]] = []
        for good, bad, subtype in pairs:
            key = (good, bad)
            if key in seen_pairs:
                continue
            good_py = to_diacritic(good)
            bad_py = to_diacritic(bad)
            flags = quality_flags(good, bad, good_py, bad_py)
            if "identical_zh" in flags or "identical_diacritic" in flags:
                continue
            selected.append((good, bad, subtype))
            seen_pairs.add(key)
            if len(selected) >= desired[phenomenon]:
                break
        if len(selected) < desired[phenomenon]:
            raise RuntimeError(f"Only {len(selected)} usable template pairs for {phenomenon}; need {desired[phenomenon]}")
        for good, bad, subtype in selected:
            items.append(
                make_item(
                    index=index,
                    phenomenon=phenomenon,
                    subtype=subtype,
                    good_zh=good,
                    bad_zh=bad,
                    data_source="hand_written_templates_no_train_data",
                    generation_method="fixed_seed_template_cartesian_product",
                )
            )
            index += 1
    rng.shuffle(items)
    for index, item in enumerate(items):
        item["id"] = f"eval4_{index:05d}"
    meta = {
        "source": "custom_template_dataset",
        "seed": seed,
        "target_items": target_items,
        "n_items": len(items),
        "phenomena": dict(Counter(item["phenomenon"] for item in items)),
        "generation_note": "Hand-written Chinese minimal-pair templates; no train split or factual QA data used.",
    }
    return items, meta


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = dict(item)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["quality_flags"] = json.dumps(row["quality_flags"], ensure_ascii=False)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output_dir = project_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.force_custom:
        items, meta = discover_existing_dataset(root, args)
    else:
        items, meta = [], None

    if items and meta:
        print("existing local Chinese BLiMP-style dataset found")
        print(f"path: {meta['path']}")
        print(f"items: {meta['n_items']}")
        print(f"fields: {', '.join(meta['fields'])}")
        print(f"categories: {json.dumps(meta['categories'], ensure_ascii=False, sort_keys=True)}")
    elif args.prefer_zhoblimp and not args.force_custom:
        print("no valid local Zho-BLMP/Chinese BLiMP-style dataset found; trying Hugging Face ZhoBLiMP")
        items, meta = load_zhoblimp_hf(args)
        if items and meta:
            print("ZhoBLiMP loaded")
            print(f"dataset: {meta['dataset']}")
            print(f"split: {meta['split']}")
            print(f"loaded items: {meta['n_loaded_items']}")
            print(f"sampled items: {meta['n_items']}")
            print(f"raw fields: {', '.join(meta['raw_fields'])}")
            print(f"categories: {json.dumps(meta['categories'], ensure_ascii=False, sort_keys=True)}")
            print(f"UID/paradigm count in loaded data: {meta['uids']}")
        else:
            print("ZhoBLiMP could not be loaded; building custom Eval 4 data")
            items, meta = build_custom_dataset(args.target_items, args.seed)
    else:
        print("no valid existing Zho-BLMP/Chinese BLiMP-style local dataset found; building custom Eval 4 data")
        items, meta = build_custom_dataset(args.target_items, args.seed)

    jsonl_path = output_dir / DEFAULT_JSONL
    review_path = output_dir / DEFAULT_REVIEW
    write_jsonl(jsonl_path, items)
    write_review_csv(review_path, items)

    by_phenomenon = dict(Counter(item["phenomenon"] for item in items))
    flag_counts: Counter[str] = Counter(flag for item in items for flag in item["quality_flags"])
    meta = {
        **(meta or {}),
        "output_jsonl": str(jsonl_path),
        "output_review_csv": str(review_path),
        "n_items": len(items),
        "phenomena": by_phenomenon,
        "quality_flag_counts": dict(flag_counts),
        "fields": FIELDS,
    }
    (output_dir / "eval4_chinese_blimp_style_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {jsonl_path}")
    print(f"wrote: {review_path}")
    print(f"dataset size: {len(items)}")
    print(f"phenomena: {json.dumps(by_phenomenon, ensure_ascii=False, sort_keys=True)}")
    if flag_counts:
        print(f"quality flags: {json.dumps(dict(flag_counts), ensure_ascii=False, sort_keys=True)}")
    else:
        print("quality flags: none")
    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.print_examples, len(items)))
    for item in sample:
        print(
            f"[{item['id']}] {item['phenomenon']} good={item['good_sentence_zh']} "
            f"bad={item['bad_sentence_zh']}"
        )


if __name__ == "__main__":
    main()
