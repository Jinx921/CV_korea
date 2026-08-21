#!/usr/bin/env python3
"""이미지·텍스트 Top-K를 결합해 보수적인 RAG 매핑 검수 후보를 만든다.

검수 전에는 어떤 항목도 matched로 확정하지 않는다. 이미지 유사도 하한을 넘고
교차 모달 근거가 있거나, 매우 강한 이미지 단독 근거가 있는 항목만 review로
보내며 나머지는 unmatched로 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "mapping" / "image_candidates" / "image_candidates.jsonl"
)
DEFAULT_TEXT_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "mapping" / "text_candidates" / "text_candidates.jsonl"
)
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mapping" / "combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-candidates", type=Path, default=DEFAULT_IMAGE_CANDIDATES
    )
    parser.add_argument("--text-candidates", type=Path, default=DEFAULT_TEXT_CANDIDATES)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-image-score", type=float, default=0.60)
    parser.add_argument("--minimum-image-margin", type=float, default=0.02)
    parser.add_argument("--strong-image-score", type=float, default=0.65)
    parser.add_argument("--strong-image-margin", type=float, default=0.05)
    parser.add_argument("--rrf-k", type=int, default=60)
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


def index_unique(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = str(row.get(key_name, ""))
        if not key:
            raise RuntimeError(f"{key_name}가 없는 행이 있습니다.")
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        raise RuntimeError(f"중복 {key_name}: {duplicates[:5]}")
    return indexed


def candidate_map(row: dict[str, Any]) -> dict[str, tuple[int, dict[str, Any]]]:
    return {
        str(candidate["doc_id"]): (rank, candidate)
        for rank, candidate in enumerate(row.get("candidates", []), start=1)
    }


def image_channel_agreement(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("dino_rank") is not None
        and candidate.get("clip_rank") is not None
    )


def choose_cross_modal_doc_id(
    image_by_doc: dict[str, tuple[int, dict[str, Any]]],
    text_by_doc: dict[str, tuple[int, dict[str, Any]]],
    rrf_k: int,
) -> tuple[str | None, float | None]:
    overlap = set(image_by_doc) & set(text_by_doc)
    if not overlap:
        return None, None
    scored = []
    for doc_id in overlap:
        image_rank = image_by_doc[doc_id][0]
        text_rank = text_by_doc[doc_id][0]
        score = 1.0 / (rrf_k + image_rank) + 1.0 / (rrf_k + text_rank)
        scored.append((score, -image_rank, -text_rank, doc_id))
    scored.sort(reverse=True)
    return scored[0][3], scored[0][0]


def main() -> int:
    args = parse_args()
    image_path = args.image_candidates.expanduser().resolve()
    text_path = args.text_candidates.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    image_rows = load_jsonl(image_path)
    text_rows = load_jsonl(text_path)
    images_by_key = index_unique(image_rows, "image_key")
    texts_by_key = index_unique(text_rows, "image_key")
    if set(images_by_key) != set(texts_by_key):
        difference = sorted(set(images_by_key) ^ set(texts_by_key))
        raise RuntimeError(f"이미지·텍스트 키가 다릅니다: {difference[:5]}")

    dataset_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        dataset_rows.extend(load_jsonl(dataset_dir / f"{split}.jsonl"))
    dataset_by_key = index_unique(
        [
            {
                **row,
                "image_key": f"{row['split']}:{row['image_name']}",
            }
            for row in dataset_rows
        ],
        "image_key",
    )
    if set(dataset_by_key) != set(images_by_key):
        difference = sorted(set(dataset_by_key) ^ set(images_by_key))
        raise RuntimeError(f"대회 데이터·후보 키가 다릅니다: {difference[:5]}")

    combined_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for image_key in sorted(
        images_by_key,
        key=lambda value: (
            ("train", "validation", "test").index(value.split(":", 1)[0]),
            value,
        ),
    ):
        image_row = images_by_key[image_key]
        text_row = texts_by_key[image_key]
        dataset_row = dataset_by_key[image_key]
        image_by_doc = candidate_map(image_row)
        text_by_doc = candidate_map(text_row)
        image_top_id = next(iter(image_by_doc), None)
        text_top_id = next(iter(text_by_doc), None)
        proposed_doc_id, rrf_score = choose_cross_modal_doc_id(
            image_by_doc, text_by_doc, args.rrf_k
        )

        cross_modal = proposed_doc_id is not None
        if proposed_doc_id is None:
            proposed_doc_id = image_top_id
        if proposed_doc_id is None:
            raise RuntimeError(f"이미지 후보가 없습니다: {image_key}")

        image_rank, image_candidate = image_by_doc[proposed_doc_id]
        text_entry = text_by_doc.get(proposed_doc_id)
        text_rank = text_entry[0] if text_entry else None
        text_candidate = text_entry[1] if text_entry else None
        image_score = float(image_candidate["late_interaction_score"])
        image_margin = float(image_row.get("top1_top2_margin") or 0.0)
        channels_agree = image_channel_agreement(image_candidate)
        cross_modal_top1 = image_top_id == text_top_id == proposed_doc_id

        if (
            cross_modal_top1
            and image_score >= args.minimum_image_score
            and image_margin >= args.minimum_image_margin
            and channels_agree
        ):
            review_priority = "high_cross_modal"
            reason = "image_text_top1_same_and_image_gate_passed"
            confidence = "high_candidate"
        elif cross_modal and image_score >= args.minimum_image_score:
            review_priority = "review_cross_modal"
            reason = "image_text_top5_overlap_and_image_score_passed"
            confidence = "medium_candidate"
        elif (
            not cross_modal
            and image_top_id == proposed_doc_id
            and image_score >= args.strong_image_score
            and image_margin >= args.strong_image_margin
            and channels_agree
        ):
            review_priority = "review_strong_image_only"
            reason = "strong_image_only_gate_passed"
            confidence = "medium_candidate"
        else:
            review_priority = None
            if image_score < args.minimum_image_score:
                reason = "image_score_below_threshold"
            elif not cross_modal:
                reason = "no_image_text_doc_id_agreement"
            elif image_margin < args.minimum_image_margin:
                reason = "image_margin_below_threshold"
            elif not channels_agree:
                reason = "dino_clip_disagreement"
            else:
                reason = "conservative_gate_rejected"
            confidence = "low"

        mapping_status = "review" if review_priority else "unmatched"
        combined = {
            "split": str(dataset_row["split"]),
            "question_id": str(dataset_row.get("question_id", "")),
            "image_name": str(dataset_row["image_name"]),
            "image_key": image_key,
            "question_form": str(dataset_row.get("question_form", "")),
            "mapping_status": mapping_status,
            # 검수 전에는 최종 doc_id를 비워 잘못된 RAG 주입을 방지한다.
            "doc_id": None,
            "title": None,
            "description": None,
            "confidence": confidence,
            "reason": reason,
            "review_priority": review_priority,
            "proposed_doc_id": proposed_doc_id if review_priority else None,
            "proposed_title": (
                str(image_candidate.get("title", "")) if review_priority else None
            ),
            "proposed_description": (
                str(image_candidate.get("description", "")) if review_priority else None
            ),
            "proposed_image_path": (
                str(image_candidate.get("image_path", "")) if review_priority else None
            ),
            "proposed_image_url": (
                str(image_candidate.get("image_url", "")) if review_priority else None
            ),
            "evidence": {
                "image_score": round(image_score, 6),
                "image_top1_top2_margin": round(image_margin, 6),
                "image_rank": image_rank,
                "dino_rank": image_candidate.get("dino_rank"),
                "clip_rank": image_candidate.get("clip_rank"),
                "dino_clip_agreement": channels_agree,
                "text_rank": text_rank,
                "text_score": (
                    text_candidate.get("bm25_score") if text_candidate else None
                ),
                "text_exact_title_in_query": (
                    text_candidate.get("exact_title_in_query")
                    if text_candidate
                    else None
                ),
                "matched_text_tokens": (
                    text_candidate.get("matched_tokens", []) if text_candidate else []
                ),
                "cross_modal_doc_id_agreement": cross_modal,
                "cross_modal_top1_agreement": cross_modal_top1,
                "rrf_score": round(rrf_score, 8) if rrf_score is not None else None,
            },
        }
        combined_rows.append(combined)

        if review_priority:
            review_rows.append(
                {
                    **combined,
                    "question": str(dataset_row.get("question", "")),
                    "options": dataset_row.get("options", []) or [],
                    "review_decision": "",
                    "review_note": "",
                }
            )

    review_rows.sort(
        key=lambda row: (
            {
                "high_cross_modal": 0,
                "review_cross_modal": 1,
                "review_strong_image_only": 2,
            }[str(row["review_priority"])],
            -float(row["evidence"]["image_score"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "rag_mapping_candidates.jsonl"
    review_jsonl_path = output_dir / "mapping_review.jsonl"
    review_csv_path = output_dir / "mapping_review.csv"
    summary_path = output_dir / "combined_mapping_summary.json"
    atomic_write_jsonl(candidates_path, combined_rows)
    atomic_write_jsonl(review_jsonl_path, review_rows)

    csv_fields = [
        "image_key",
        "split",
        "question_id",
        "image_name",
        "question_form",
        "review_priority",
        "proposed_doc_id",
        "proposed_title",
        "question",
        "options",
        "proposed_description",
        "proposed_image_path",
        "proposed_image_url",
        "image_score",
        "image_margin",
        "image_rank",
        "dino_rank",
        "clip_rank",
        "text_rank",
        "text_score",
        "matched_text_tokens",
        "review_decision",
        "review_note",
    ]
    temporary_csv = review_csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in review_rows:
            evidence = row["evidence"]
            writer.writerow(
                {
                    "image_key": row["image_key"],
                    "split": row["split"],
                    "question_id": row["question_id"],
                    "image_name": row["image_name"],
                    "question_form": row["question_form"],
                    "review_priority": row["review_priority"],
                    "proposed_doc_id": row["proposed_doc_id"],
                    "proposed_title": row["proposed_title"],
                    "question": row["question"],
                    "options": json.dumps(row["options"], ensure_ascii=False),
                    "proposed_description": row["proposed_description"],
                    "proposed_image_path": row["proposed_image_path"],
                    "proposed_image_url": row["proposed_image_url"],
                    "image_score": evidence["image_score"],
                    "image_margin": evidence["image_top1_top2_margin"],
                    "image_rank": evidence["image_rank"],
                    "dino_rank": evidence["dino_rank"],
                    "clip_rank": evidence["clip_rank"],
                    "text_rank": evidence["text_rank"],
                    "text_score": evidence["text_score"],
                    "matched_text_tokens": json.dumps(
                        evidence["matched_text_tokens"], ensure_ascii=False
                    ),
                    "review_decision": "",
                    "review_note": "",
                }
            )
    temporary_csv.replace(review_csv_path)

    status_counts = Counter(row["mapping_status"] for row in combined_rows)
    priority_counts = Counter(
        str(row["review_priority"])
        for row in review_rows
        if row["review_priority"]
    )
    summary = {
        "record_count": len(combined_rows),
        "status_counts": dict(status_counts),
        "review_priority_counts": dict(priority_counts),
        "thresholds": {
            "minimum_image_score": args.minimum_image_score,
            "minimum_image_margin": args.minimum_image_margin,
            "strong_image_score": args.strong_image_score,
            "strong_image_margin": args.strong_image_margin,
            "rrf_k": args.rrf_k,
        },
        "safety_policy": {
            "matched_before_review": 0,
            "unmatched_has_null_doc_id": True,
            "review_requires_manual_yes_or_no": True,
        },
        "outputs": {
            "candidate_db": str(candidates_path),
            "review_jsonl": str(review_jsonl_path),
            "review_csv": str(review_csv_path),
        },
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
