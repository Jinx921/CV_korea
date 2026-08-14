#!/usr/bin/env python3
"""대회 질문 텍스트와 TourAPI 제목을 대조해 상세 수집 후보를 만든다.

이 단계는 API를 호출하지 않는다. train은 질문·선택지·정답을 사용하고,
validation/test는 평가 누수를 피하기 위해 질문·선택지만 사용한다.
텍스트에 직접 나타나지 않는 대상은 이후 이미지 유사도 후보와 합친다.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "processed"
DEFAULT_TOURAPI_INPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "area_based"
    / "image_candidates.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "targets"

GENERIC_ALIASES = {
    "한국",
    "대한민국",
    "우리나라",
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "제주",
    "공원",
    "시장",
    "박물관",
    "미술관",
    "전시관",
    "문화원",
    "문화관",
    "축제",
    "한옥",
    "사찰",
    "학교",
    "도서관",
    "관광지",
    "시스템",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="대회 문항 텍스트에 직접 등장하는 TourAPI 항목을 선별합니다."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"전처리된 split JSONL 폴더. 기본값: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--tourapi-input",
        type=Path,
        default=DEFAULT_TOURAPI_INPUT,
        help=f"TourAPI 이미지 후보 JSONL. 기본값: {DEFAULT_TOURAPI_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 폴더. 기본값: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--ocr-input",
        type=Path,
        default=None,
        help="선택 사항: split, image_name, ocr_text가 있는 OCR JSONL",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_no}가 JSON 객체가 아닙니다.")
            rows.append(value)
    return rows


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(re.findall(r"[가-힣a-z0-9]+", text))


def title_aliases(title: str) -> set[str]:
    raw_aliases = {title}
    raw_aliases.add(re.split(r"[\(\[\{]", title, maxsplit=1)[0])
    raw_aliases.update(re.split(r"[/·|]", title))

    aliases = {
        normalized
        for raw in raw_aliases
        if (normalized := normalize(raw))
        and len(normalized) >= 2
        and normalized not in GENERIC_ALIASES
    }
    return aliases


def record_text(record: dict[str, Any], ocr_text: str = "") -> str:
    parts = [str(record.get("question", ""))]
    parts.extend(str(option) for option in record.get("options", []) or [])
    # train 정답만 후보 발견에 이용한다. validation/test 정답은 사용하지 않는다.
    if record.get("split") == "train" and record.get("answer") is not None:
        parts.append(str(record["answer"]))
    if ocr_text:
        parts.append(ocr_text)
    return normalize(" ".join(parts))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    tourapi_path = args.tourapi_input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    dataset_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        dataset_rows.extend(load_jsonl(dataset_dir / f"{split}.jsonl"))
    candidates = load_jsonl(tourapi_path)
    ocr_by_image: dict[tuple[str, str], str] = {}
    if args.ocr_input is not None:
        for ocr_row in load_jsonl(args.ocr_input.expanduser().resolve()):
            key = (
                str(ocr_row.get("split", "")),
                str(ocr_row.get("image_name", "")),
            )
            ocr_by_image[key] = str(ocr_row.get("ocr_text", ""))

    alias_to_content_ids: dict[str, set[str]] = defaultdict(set)
    content_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        content_id = str(candidate.get("contentid", "")).strip()
        title = str(candidate.get("title", "")).strip()
        if not content_id or not title:
            continue
        content_by_id[content_id] = candidate
        for alias in title_aliases(title):
            alias_to_content_ids[alias].add(content_id)

    aliases_by_length: dict[int, set[str]] = defaultdict(set)
    for alias in alias_to_content_ids:
        aliases_by_length[len(alias)].add(alias)

    broad_evidence_by_content: dict[str, list[dict[str, str]]] = defaultdict(list)
    high_confidence_ids: set[str] = set()
    matched_question_ids: set[str] = set()
    broadly_matched_question_ids: set[str] = set()
    for record in dataset_rows:
        image_key = (
            str(record.get("split", "")),
            str(record.get("image_name", "")),
        )
        text = record_text(record, ocr_by_image.get(image_key, ""))
        matched_aliases: set[str] = set()
        for length, aliases in aliases_by_length.items():
            if length > len(text):
                continue
            ngrams = {
                text[start : start + length]
                for start in range(0, len(text) - length + 1)
            }
            matched_aliases.update(ngrams & aliases)

        if not matched_aliases:
            continue
        global_id = str(record.get("global_id", ""))
        broadly_matched_question_ids.add(global_id)
        for alias in sorted(matched_aliases):
            for content_id in alias_to_content_ids[alias]:
                tier = "direct_3plus" if len(alias) >= 3 else "short_2char"
                if tier == "direct_3plus":
                    high_confidence_ids.add(content_id)
                    matched_question_ids.add(global_id)
                broad_evidence_by_content[content_id].append(
                    {
                        "global_id": global_id,
                        "split": str(record.get("split", "")),
                        "image_name": str(record.get("image_name", "")),
                        "matched_alias": alias,
                        "match_tier": tier,
                    }
                )

    target_rows: list[dict[str, Any]] = []
    for content_id in sorted(high_confidence_ids, key=lambda value: int(value)):
        candidate = dict(content_by_id[content_id])
        candidate["selection_method"] = "dataset_text_title_match"
        candidate["candidate_tier"] = "high"
        candidate["match_evidence"] = broad_evidence_by_content[content_id]
        target_rows.append(candidate)

    broad_rows: list[dict[str, Any]] = []
    for content_id in sorted(broad_evidence_by_content, key=lambda value: int(value)):
        candidate = dict(content_by_id[content_id])
        candidate["selection_method"] = "dataset_text_ocr_broad_match"
        candidate["candidate_tier"] = (
            "high" if content_id in high_confidence_ids else "broad_only"
        )
        candidate["match_evidence"] = broad_evidence_by_content[content_id]
        broad_rows.append(candidate)

    target_path = output_dir / "text_target_candidates.jsonl"
    broad_path = output_dir / "broad_image_candidates.jsonl"
    atomic_write_jsonl(target_path, target_rows)
    atomic_write_jsonl(broad_path, broad_rows)
    summary = {
        "dataset_record_count": len(dataset_rows),
        "tourapi_image_candidate_count": len(candidates),
        "matched_dataset_record_count": len(matched_question_ids),
        "unique_tourapi_target_count": len(target_rows),
        "broadly_matched_dataset_record_count": len(broadly_matched_question_ids),
        "broad_image_candidate_count": len(broad_rows),
        "ocr_record_count": len(ocr_by_image),
        "note": (
            "text_target_candidates는 3글자 이상 고신뢰 후보이고, "
            "broad_image_candidates는 두 글자 약한 일치까지 포함한 이미지 "
            "검증용 후보입니다."
        ),
        "output": str(target_path),
        "broad_output": str(broad_path),
    }
    atomic_write_json(output_dir / "text_target_summary.json", summary)

    print(f"대회 문항                 : {len(dataset_rows):,}건")
    print(f"텍스트로 후보가 발견된 문항: {len(matched_question_ids):,}건")
    print(f"고신뢰 TourAPI 후보        : {len(target_rows):,}건")
    print(f"광범위 이미지 후보         : {len(broad_rows):,}건")
    print(f"고신뢰 후보 JSONL          : {target_path}")
    print(f"광범위 후보 JSONL          : {broad_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
