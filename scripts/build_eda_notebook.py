from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "01_eda_preprocessing.ipynb"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (.venv AI_CV)",
        "language": "python",
        "name": "ai-cv",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# 한국문화 멀티모달 질의응답 데이터: 전처리 및 EDA

이 노트북은 국립국어원 인공지능(AI)말평의 **한국문화 멀티모달 질의응답** 데이터 2,000문항을 점검한다.

목표:

1. JSON 스키마와 이미지 연결 무결성 검증
2. split·문항 유형·텍스트 길이·답안 형식 분석
3. 이미지 크기·해상도·파일 형식 분석
4. 반복 질문 템플릿과 교차 split 동일 이미지 점검
5. 원본을 수정하지 않는 정규화 파생 데이터 생성

> 주의: 질문 키워드 기반 OCR/부정형/복수정답 표시는 휴리스틱이며 정답 라벨이 아니다. 자동 탐지 결과는 삭제 기준이 아니라 수동 검토 큐로 사용한다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """import os
import sys
import json
import re
import hashlib
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from IPython.display import Markdown, display

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PROCESSED_ROOT = OUTPUT_ROOT / "processed"
EDA_ROOT = OUTPUT_ROOT / "eda"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
EDA_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
FORMS = ["MC", "SA", "LA"]
EXPECTED_COUNTS = {"train": 1000, "validation": 200, "test": 800}
COMPUTE_SHA256 = True  # 8.4 GiB 전체를 읽는다. 빠른 재실행 시 False로 변경 가능

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_rows", 100)
sns.set_theme(style="whitegrid", context="notebook")
print(f"Python: {sys.version.split()[0]}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data root exists: {DATA_ROOT.exists()}")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. 원본 JSON 탐색과 기본 구조 확인

폴더명이 macOS에서 NFD 유니코드로 저장되어 있을 수 있으므로 한글 폴더명을 직접 하드코딩하지 않고 split 접미사로 JSON을 탐색한다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def find_split_json(split: str) -> Path:
    candidates = sorted(DATA_ROOT.rglob(f"*_{split}.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one JSON for {split}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


json_paths = {split: find_split_json(split) for split in SPLITS}
raw_by_split = {
    split: json.loads(path.read_text(encoding="utf-8"))
    for split, path in json_paths.items()
}

overview = pd.DataFrame(
    [
        {
            "split": split,
            "json_path": str(json_paths[split].relative_to(PROJECT_ROOT)),
            "records": len(raw_by_split[split]),
            "expected": EXPECTED_COUNTS[split],
            "count_match": len(raw_by_split[split]) == EXPECTED_COUNTS[split],
        }
        for split in SPLITS
    ]
)
display(overview)
display(raw_by_split["train"][0])"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. 손실 없는 텍스트 정규화와 평탄화

- 모든 문자열을 Unicode NFC로 통일한다.
- 앞뒤 공백과 연속 공백을 정리하되 의미가 있는 문장부호는 유지한다.
- 원문 답안은 유지하고 MC 선택지 번호 집합을 별도 파생 변수로 만든다.
- test에는 공개 정답이 없으므로 `answer=None`으로 둔다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """MC_CHOICE_RE = re.compile(r"(?<!\\d)([1-5])(?!\\d)")
LENGTH_RE = re.compile(r"(?P<length>\\d+)\\s*(?P<unit>음절|글자|어절|자)(?:로|으로)?")
OCR_RE = re.compile(r"적혀|쓰여|써 있|문구|글자|안내문|안내판|표지판|간판|현수막|포스터|메뉴판|명판|읽(?:고|어|으)")
NEGATION_RE = re.compile(r"옳지 않|틀린|아닌|않은|제외|수 없는|해당하지 않")
MULTI_RE = re.compile(r"모두\\s*(?:고르|답|찾)|전부\\s*(?:고르|답|찾)|각각|복수")


def normalize_text(value) -> str:
    text = unicodedata.normalize("NFC", "" if value is None else str(value))
    return re.sub(r"\\s+", " ", text).strip()


def visible_char_count(text: str) -> int:
    return sum(char.isalnum() for char in text)


def extract_mc_choices(answer: str) -> list[int]:
    return sorted({int(choice) for choice in MC_CHOICE_RE.findall(answer or "")})


def parse_length_constraint(question: str):
    match = LENGTH_RE.search(question)
    if not match:
        return None, None
    return int(match.group("length")), match.group("unit")


rows = []
schema_issues = []
for split in SPLITS:
    for position, raw in enumerate(raw_by_split[split]):
        metadata = raw.get("metadata", {})
        model_input = raw.get("model_input", {})
        model_output = raw.get("model_output")
        question_id = normalize_text(metadata.get("question_id"))
        question = normalize_text(model_input.get("question"))
        image_name = normalize_text(model_input.get("image_name"))
        question_form = normalize_text(metadata.get("question_form")).upper()
        options = [normalize_text(option) for option in (model_input.get("options") or [])]
        answer = None if model_output is None else normalize_text(model_output.get("answer"))
        requested_length, requested_unit = parse_length_constraint(question)

        row = {
            "global_id": f"{split}:{question_id}",
            "question_id": question_id,
            "split": split,
            "question_form": question_form,
            "image_name": image_name,
            "image_path": DATA_ROOT / split / image_name,
            "question": question,
            "options": options,
            "option_count": len(options),
            "answer": answer,
            "mc_choices": extract_mc_choices(answer) if question_form == "MC" else [],
            "question_chars": len(question),
            "question_eojeol": len(question.split()),
            "answer_chars": None if answer is None else len(answer),
            "answer_visible_chars": None if answer is None else visible_char_count(answer),
            "requested_length": requested_length,
            "requested_unit": requested_unit,
            "ocr_cue": bool(OCR_RE.search(question)),
            "negation_cue": bool(NEGATION_RE.search(question)),
            "multi_answer_cue": bool(MULTI_RE.search(question)),
        }
        rows.append(row)

        if metadata.get("split") != split:
            schema_issues.append((row["global_id"], "split_mismatch"))
        if question_form not in FORMS:
            schema_issues.append((row["global_id"], "unknown_question_form"))
        if question_form == "MC" and len(options) != 5:
            schema_issues.append((row["global_id"], "mc_option_count_not_5"))
        if question_form != "MC" and options:
            schema_issues.append((row["global_id"], "non_mc_has_options"))
        if split != "test" and answer is None:
            schema_issues.append((row["global_id"], "missing_answer"))
        if split == "test" and answer is not None:
            schema_issues.append((row["global_id"], "test_answer_present"))

df = pd.DataFrame(rows)
print(f"Records: {len(df):,}")
print(f"Schema issues: {len(schema_issues):,}")
display(df.head(3))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## 3. split 및 문항 유형 분포")
)

cells.append(
    nbf.v4.new_code_cell(
        """split_form = (
    df.groupby(["split", "question_form"])
      .size()
      .unstack(fill_value=0)
      .reindex(index=SPLITS, columns=FORMS, fill_value=0)
)
display(split_form)

plot_df = split_form.reset_index().melt(
    id_vars="split", var_name="question_form", value_name="count"
)
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.barplot(
    data=plot_df, x="split", y="count", hue="question_form",
    order=SPLITS, hue_order=FORMS, palette="colorblind", ax=ax
)
for container in ax.containers:
    ax.bar_label(container, padding=2, fontsize=9)
ax.set(title="Question mix by split", xlabel="Split", ylabel="Questions")
ax.legend(title="Form")
plt.tight_layout()
fig.savefig(EDA_ROOT / "question_mix.png", dpi=180, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## 4. 질문 및 정답 길이")
)

cells.append(
    nbf.v4.new_code_cell(
        """labeled = df[df["answer"].notna()].copy()

text_summary = pd.DataFrame(
    {
        "question_chars": df["question_chars"].describe(percentiles=[0.25, 0.5, 0.75, 0.95]),
        "answer_chars_labeled": labeled["answer_chars"].describe(percentiles=[0.25, 0.5, 0.75, 0.95]),
    }
)
display(text_summary)

answer_by_form = (
    labeled.groupby("question_form")["answer_chars"]
    .describe(percentiles=[0.5, 0.95])
    .reindex(FORMS)
)
display(answer_by_form)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(
    data=df, x="question_form", y="question_chars", order=FORMS,
    hue="question_form", palette="colorblind", legend=False,
    showfliers=False, ax=axes[0]
)
axes[0].set(title="Question length", xlabel="Form", ylabel="Characters (outliers hidden)")
sns.boxplot(
    data=labeled, x="question_form", y="answer_chars", order=FORMS,
    hue="question_form", palette="colorblind", legend=False,
    showfliers=False, ax=axes[1]
)
axes[1].set(title="Answer length: train + validation", xlabel="Form", ylabel="Characters (outliers hidden)")
plt.tight_layout()
fig.savefig(EDA_ROOT / "text_lengths.png", dpi=180, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. 이미지 무결성·크기·정확 중복

이미지 헤더에서 크기·형식·모드를 읽고, `COMPUTE_SHA256=True`일 때 전체 파일 바이트의 SHA-256을 계산한다. SHA-256 중복은 완전히 같은 파일만 탐지하며, 크롭·리사이즈·압축률이 다른 근접 중복은 탐지하지 않는다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_image(item):
    global_id, path = item
    result = {
        "global_id": global_id,
        "image_found": path.is_file(),
        "image_bytes": None,
        "image_mb": None,
        "image_width": None,
        "image_height": None,
        "image_megapixels": None,
        "image_aspect_ratio": None,
        "image_format": None,
        "image_mode": None,
        "image_sha256": None,
        "image_error": None,
    }
    if not path.is_file():
        return result
    try:
        size = path.stat().st_size
        with Image.open(path) as image:
            width, height = image.size
            result.update(
                image_bytes=size,
                image_mb=size / (1024 * 1024),
                image_width=width,
                image_height=height,
                image_megapixels=width * height / 1_000_000,
                image_aspect_ratio=width / height if height else None,
                image_format=image.format,
                image_mode=image.mode,
            )
        if COMPUTE_SHA256:
            result["image_sha256"] = sha256_file(path)
    except Exception as exc:
        result["image_error"] = f"{type(exc).__name__}: {exc}"
    return result


image_items = list(df[["global_id", "image_path"]].itertuples(index=False, name=None))
with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
    image_profiles = list(executor.map(profile_image, image_items))

image_df = pd.DataFrame(image_profiles)
df = df.merge(image_df, on="global_id", how="left", validate="one_to_one")

image_summary = df.groupby("split").agg(
    images=("image_found", "size"),
    missing=("image_found", lambda s: (~s).sum()),
    total_gib=("image_bytes", lambda s: s.sum() / 1024**3),
    median_mb=("image_mb", "median"),
    p95_mb=("image_mb", lambda s: s.quantile(0.95)),
    max_mb=("image_mb", "max"),
    median_mp=("image_megapixels", "median"),
    max_mp=("image_megapixels", "max"),
).round(3)
display(image_summary)
display(df["image_format"].value_counts(dropna=False).rename("count").to_frame())

fig, ax = plt.subplots(figsize=(9, 5))
plot_images = df.dropna(subset=["image_megapixels", "image_mb"])
sns.scatterplot(
    data=plot_images, x="image_megapixels", y="image_mb", hue="split",
    hue_order=SPLITS, palette="colorblind", alpha=0.6, s=30, ax=ax
)
ax.set(
    xlim=(0, plot_images["image_megapixels"].quantile(0.99) * 1.05),
    ylim=(0, plot_images["image_mb"].quantile(0.99) * 1.05),
    title="Image compute profile (axes capped at p99)",
    xlabel="Megapixels", ylabel="File size (MiB)",
)
ax.legend(title="Split")
plt.tight_layout()
fig.savefig(EDA_ROOT / "image_profile.png", dpi=180, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. 반복 질문 문구와 교차 split 동일 이미지

동일한 질문 문구는 일반적인 MC 템플릿 반복일 수 있으므로 곧바로 데이터 누수로 해석하지 않는다. 반면 동일 이미지 해시가 train/validation/test 사이에 나타나는 경우, 같은 시각 정보에 대한 다른 질문이 존재할 수 있어 평가 해석과 retrieval cache 설계에 영향을 준다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def cross_split_groups(frame: pd.DataFrame, key: str):
    groups = []
    for value, group in frame.dropna(subset=[key]).groupby(key):
        if group["split"].nunique() > 1:
            groups.append((value, group.copy()))
    return groups


question_text_groups = cross_split_groups(df, "question")
image_name_groups = cross_split_groups(df, "image_name")
image_hash_groups = cross_split_groups(df, "image_sha256") if COMPUTE_SHA256 else []

duplicate_overview = pd.DataFrame(
    [
        {"check": "Repeated exact question text", "cross_split_groups": len(question_text_groups), "affected_rows": sum(len(g) for _, g in question_text_groups)},
        {"check": "Repeated image filename", "cross_split_groups": len(image_name_groups), "affected_rows": sum(len(g) for _, g in image_name_groups)},
        {"check": "Identical image SHA-256", "cross_split_groups": len(image_hash_groups), "affected_rows": sum(len(g) for _, g in image_hash_groups)},
    ]
)
display(duplicate_overview)

if image_hash_groups:
    duplicate_image_rows = pd.concat(
        [group.assign(duplicate_hash=hash_value) for hash_value, group in image_hash_groups],
        ignore_index=True,
    )
    display(
        duplicate_image_rows[
            ["global_id", "split", "image_name", "question_form", "question", "answer", "duplicate_hash"]
        ].sort_values("duplicate_hash")
    )

largest_templates = sorted(
    [
        {
            "question": text,
            "rows": len(group),
            "splits": ", ".join(sorted(group["split"].unique())),
        }
        for text, group in question_text_groups
    ],
    key=lambda item: item["rows"], reverse=True,
)[:10]
display(pd.DataFrame(largest_templates))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## 7. 답안 형식 및 품질 위험")
)

cells.append(
    nbf.v4.new_code_cell(
        """def constraint_matches(row):
    if row["answer"] is None or pd.isna(row["requested_length"]):
        return None
    observed = len(row["answer"].split()) if row["requested_unit"] == "어절" else visible_char_count(row["answer"])
    return observed == int(row["requested_length"])


df["length_constraint_match"] = df.apply(constraint_matches, axis=1)
labeled_constraints = df[df["answer"].notna() & df["requested_length"].notna()].copy()
la_over_250 = df[(df["question_form"] == "LA") & (df["answer_chars"] > 250)].copy()
large_images = df[df["image_mb"] > 15].copy()
high_resolution = df[df["image_megapixels"] > 25].copy()

mc_choice_frequency = Counter(
    choice
    for choices in labeled.loc[labeled["question_form"] == "MC", "mc_choices"]
    for choice in choices
)
mc_summary = pd.DataFrame(
    {"choice": range(1, 6), "frequency": [mc_choice_frequency.get(i, 0) for i in range(1, 6)]}
)
display(mc_summary)

quality_summary = pd.DataFrame(
    [
        {"check": "Missing image", "count": int((~df["image_found"]).sum())},
        {"check": "Image open error", "count": int(df["image_error"].notna().sum())},
        {"check": "Image > 15 MiB", "count": len(large_images)},
        {"check": "Image > 25 megapixels", "count": len(high_resolution)},
        {"check": "LA reference answer > 250 chars", "count": len(la_over_250)},
        {"check": "Answer/requested-length mismatch (heuristic)", "count": int((labeled_constraints["length_constraint_match"] == False).sum())},
        {"check": "Cross-split identical image hash groups", "count": len(image_hash_groups)},
    ]
)
display(quality_summary)

if len(la_over_250):
    display(la_over_250[["global_id", "question", "answer_chars", "answer"]].sort_values("answer_chars", ascending=False).head(10))

if len(labeled_constraints):
    mismatch_view = labeled_constraints[labeled_constraints["length_constraint_match"] == False]
    display(mismatch_view[["global_id", "question", "requested_length", "requested_unit", "answer"]].head(20))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 8. 질문 문구 기반 난이도 신호

OCR 필요도는 실제 이미지를 보지 않고 질문 문구만으로 추정하므로 하한에 가깝다. 수작업 태깅용 표본을 만들 때 우선순위를 정하는 용도로만 사용한다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """cue_columns = ["ocr_cue", "negation_cue", "multi_answer_cue"]
cue_labels = {
    "ocr_cue": "OCR cue",
    "negation_cue": "Negation cue",
    "multi_answer_cue": "Multi-answer cue",
}

cue_summary = pd.DataFrame(
    [
        {
            "cue": cue_labels[cue],
            "count": int(df[cue].sum()),
            "ratio_pct": round(df[cue].mean() * 100, 1),
        }
        for cue in cue_columns
    ]
)
display(cue_summary)

cue_by_form = pd.DataFrame(
    [
        {
            "question_form": form,
            "cue": cue_labels[cue],
            "ratio_pct": df.loc[df["question_form"] == form, cue].mean() * 100,
        }
        for form in FORMS
        for cue in cue_columns
    ]
)
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.barplot(
    data=cue_by_form, x="ratio_pct", y="cue", hue="question_form",
    hue_order=FORMS, palette="colorblind", ax=ax
)
ax.set(title="Question-language risk cues", xlabel="Share within form (%)", ylabel="Heuristic cue")
ax.legend(title="Form")
plt.tight_layout()
fig.savefig(EDA_ROOT / "question_cues.png", dpi=180, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 9. 전처리 파생본과 수동 검토 큐 저장

원본 JSON과 이미지는 수정하지 않는다. 아래 출력은 `outputs/`에 저장되며 공식 제출 파일이 아니라 모델 개발용 인덱스다."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """quality_issue_rows = []

def add_issues(mask, issue, detail_column=None):
    columns = ["global_id", "split", "question_id"] + ([detail_column] if detail_column else [])
    for record in df.loc[mask, columns].to_dict("records"):
        quality_issue_rows.append(
            {
                "global_id": record["global_id"],
                "split": record["split"],
                "question_id": record["question_id"],
                "issue": issue,
                "detail": record.get(detail_column, "") if detail_column else "",
            }
        )


add_issues(~df["image_found"], "missing_image")
add_issues(df["image_error"].notna(), "image_open_error", "image_error")
add_issues(df["image_mb"] > 15, "large_image_over_15mib", "image_mb")
add_issues(df["image_megapixels"] > 25, "high_resolution_over_25mp", "image_megapixels")
add_issues((df["question_form"] == "LA") & (df["answer_chars"] > 250), "la_reference_over_250_chars", "answer_chars")
add_issues(df["length_constraint_match"] == False, "answer_length_constraint_mismatch")

for hash_value, group in image_hash_groups:
    ids = " | ".join(group["global_id"])
    for record in group[["global_id", "split", "question_id"]].to_dict("records"):
        quality_issue_rows.append({**record, "issue": "cross_split_identical_image", "detail": ids})

issues_df = pd.DataFrame(quality_issue_rows).sort_values(["issue", "split", "question_id"])
issues_df.to_csv(EDA_ROOT / "quality_issues.csv", index=False, encoding="utf-8-sig")

export_columns = [
    "global_id", "question_id", "split", "question_form", "image_name",
    "question", "options", "answer", "mc_choices", "requested_length", "requested_unit",
    "ocr_cue", "negation_cue", "multi_answer_cue", "image_width", "image_height",
    "image_megapixels", "image_mb", "image_format", "image_mode", "image_sha256",
]

for split in SPLITS:
    output_path = PROCESSED_ROOT / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in df.loc[df["split"] == split, export_columns].to_dict("records"):
            clean = {
                key: (None if not isinstance(value, (list, dict)) and pd.isna(value) else value)
                for key, value in record.items()
            }
            handle.write(json.dumps(clean, ensure_ascii=False) + "\\n")

summary = {
    "records": int(len(df)),
    "split_counts": {k: int(v) for k, v in df["split"].value_counts().reindex(SPLITS).items()},
    "form_counts": {k: int(v) for k, v in df["question_form"].value_counts().reindex(FORMS).items()},
    "total_image_gib": round(float(df["image_bytes"].sum() / 1024**3), 4),
    "missing_images": int((~df["image_found"]).sum()),
    "question_chars_median": float(df["question_chars"].median()),
    "answer_chars_median_by_form": {
        form: float(labeled.loc[labeled["question_form"] == form, "answer_chars"].median())
        for form in FORMS
    },
    "ocr_cue_count": int(df["ocr_cue"].sum()),
    "negation_cue_count": int(df["negation_cue"].sum()),
    "multi_answer_cue_count": int(df["multi_answer_cue"].sum()),
    "cross_split_repeated_question_groups": int(len(question_text_groups)),
    "cross_split_identical_image_groups": int(len(image_hash_groups)),
    "large_images_over_15mib": int(len(large_images)),
    "high_resolution_over_25mp": int(len(high_resolution)),
    "la_reference_over_250_chars": int(len(la_over_250)),
    "length_constraint_mismatches": int((labeled_constraints["length_constraint_match"] == False).sum()),
    "quality_issue_rows": int(len(issues_df)),
}
(EDA_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Processed JSONL: {PROCESSED_ROOT}")
print(f"EDA summary/issues: {EDA_ROOT}")
display(pd.Series(summary, name="value").to_frame())"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## 10. 분석 결론과 다음 실험")
)

cells.append(
    nbf.v4.new_code_cell(
        """display(Markdown(f''' 
### 핵심 결론

- 데이터는 총 **{summary['records']:,}문항**, 이미지 **{summary['total_image_gib']:.2f} GiB**이며 이미지 누락은 **{summary['missing_images']}개**다.
- 문항 유형은 MC **{summary['form_counts']['MC']:,}**, SA **{summary['form_counts']['SA']:,}**, LA **{summary['form_counts']['LA']:,}**로 split 사이 비율이 거의 유지된다.
- 질문 문구상 OCR 신호가 있는 문항은 **{summary['ocr_cue_count']:,}개**지만 실제 OCR 의존도는 이보다 높을 수 있다.
- 부정형 신호가 **{summary['negation_cue_count']:,}개**, 복수 정답 신호가 **{summary['multi_answer_cue_count']:,}개**이므로 답안 라우터와 형식 검증이 중요하다.
- 교차 split 동일 이미지 SHA-256이 **{summary['cross_split_identical_image_groups']}그룹** 존재한다. 같은 이미지에 서로 다른 질문이 연결되어 있어 오류 분석과 retrieval cache 설계에서 별도로 추적해야 한다.
- 15 MiB 초과 이미지 **{summary['large_images_over_15mib']}개**, 25MP 초과 이미지 **{summary['high_resolution_over_25mp']}개**가 있어 VLM용 리사이즈와 OCR용 고해상도 파생본을 분리해야 한다.
- 공개 LA 정답 중 250자를 넘는 사례가 **{summary['la_reference_over_250_chars']}개**다. 학습 라벨은 보존하되 제출 생성 단계에서는 공식 250자 제한을 강제해야 한다.

### 다음 실험 우선순위

1. 로컬 VLM direct-answer baseline과 공식 metric scorer 구축
2. train 200개를 `visual-only / OCR / culture-knowledge / mixed`로 수작업 태깅
3. OCR 추가 전후의 validation slice 성능 비교
4. 외부 문화지식 RAG는 baseline 오류가 집중되는 문화 범주부터 구축
5. MC·SA·LA별 출력 validator를 적용해 지식 오류와 형식 오류를 분리
'''))"""
    )
)

nb["cells"] = cells
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
