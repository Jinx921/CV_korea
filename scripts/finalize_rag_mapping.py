#!/usr/bin/env python3
"""수동 검수 CSV를 반영해 최종 대회 이미지↔RAG 문서 매핑 DB를 만든다.

`mapping_review.csv`의 `review_decision`을 모두 채운 뒤 실행한다. 승인된
후보만 matched로 확정하며, 거절되거나 후보가 없던 문항은 RAG 필드를 null로
유지한다. 일부 검수가 비어 있으면 기본적으로 실패해 미검수 후보의 우발적인
사용을 막는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED_DIR = PROJECT_ROOT / "outputs" / "mapping" / "combined"
DEFAULT_CANDIDATES = DEFAULT_COMBINED_DIR / "rag_mapping_candidates.jsonl"
DEFAULT_REVIEW_CSV = DEFAULT_COMBINED_DIR / "mapping_review.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_COMBINED_DIR / "final"

YES_VALUES = {"1", "y", "yes", "true", "match", "matched", "승인", "맞음", "예"}
NO_VALUES = {"0", "n", "no", "false", "reject", "rejected", "거절", "아님", "아니오"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="검수하지 않은 review 후보를 unmatched로 처리합니다.",
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


def normalize_decision(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in YES_VALUES:
        return "yes"
    if normalized in NO_VALUES:
        return "no"
    if not normalized:
        return "pending"
    raise RuntimeError(
        f"지원하지 않는 review_decision={value!r}. yes/no 또는 승인/거절을 사용하세요."
    )


def load_review_csv(path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_key", "proposed_doc_id", "review_decision"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"검수 CSV 필수 열이 없습니다: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            image_key = str(row.get("image_key", "")).strip()
            if not image_key:
                raise RuntimeError(f"검수 CSV {line_no}행에 image_key가 없습니다.")
            if image_key in indexed:
                raise RuntimeError(f"검수 CSV에 image_key가 중복됩니다: {image_key}")
            indexed[image_key] = {str(key): str(value or "") for key, value in row.items()}
    return indexed


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    """후보 감사 정보는 유지하되 최종 RAG 필드를 명시적으로 구성한다."""

    return {
        "split": row["split"],
        "question_id": row["question_id"],
        "image_name": row["image_name"],
        "image_key": row["image_key"],
        "question_form": row.get("question_form", ""),
        "mapping_status": row["mapping_status"],
        "doc_id": row.get("doc_id"),
        "source": row.get("source"),
        "title": row.get("title"),
        "description": row.get("description"),
        "rag_image_path": row.get("rag_image_path"),
        "rag_image_url": row.get("rag_image_url"),
        "confidence": row.get("confidence", "low"),
        "reason": row.get("reason", ""),
        "review_priority": row.get("review_priority"),
        "review_decision": row.get("review_decision"),
        "review_note": row.get("review_note", ""),
        "evidence": row.get("evidence", {}),
    }


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates.expanduser().resolve()
    review_csv_path = args.review_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    candidate_rows = load_jsonl(candidates_path)
    review_by_key = load_review_csv(review_csv_path)
    candidate_keys = [str(row.get("image_key", "")) for row in candidate_rows]
    if not all(candidate_keys) or len(candidate_keys) != len(set(candidate_keys)):
        raise RuntimeError("후보 DB의 image_key가 비어 있거나 중복됩니다.")

    expected_review_keys = {
        str(row["image_key"])
        for row in candidate_rows
        if row.get("mapping_status") == "review"
    }
    supplied_review_keys = set(review_by_key)
    if expected_review_keys != supplied_review_keys:
        missing = sorted(expected_review_keys - supplied_review_keys)
        extra = sorted(supplied_review_keys - expected_review_keys)
        raise RuntimeError(
            f"후보 DB와 검수 CSV의 항목이 다릅니다. missing={missing[:5]}, extra={extra[:5]}"
        )

    pending_keys = [
        key
        for key, row in review_by_key.items()
        if normalize_decision(row["review_decision"]) == "pending"
    ]
    if pending_keys and not args.allow_pending:
        raise RuntimeError(
            f"검수하지 않은 후보가 {len(pending_keys)}건 있습니다. "
            "mapping_review.csv의 review_decision을 모두 yes/no로 채우세요. "
            f"예: {pending_keys[:5]}"
        )

    final_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        row = dict(candidate)
        image_key = str(row["image_key"])
        if row.get("mapping_status") != "review":
            row.update(
                {
                    "mapping_status": "unmatched",
                    "doc_id": None,
                    "source": None,
                    "title": None,
                    "description": None,
                    "rag_image_path": None,
                    "rag_image_url": None,
                    "review_decision": None,
                    "review_note": "",
                }
            )
            final_rows.append(canonical_row(row))
            continue

        review = review_by_key[image_key]
        decision = normalize_decision(review["review_decision"])
        csv_doc_id = review["proposed_doc_id"].strip()
        proposed_doc_id = str(row.get("proposed_doc_id") or "")
        if csv_doc_id != proposed_doc_id:
            raise RuntimeError(
                f"{image_key}: 검수 CSV의 proposed_doc_id가 원본 후보와 다릅니다. "
                f"csv={csv_doc_id}, candidate={proposed_doc_id}"
            )

        note = review.get("review_note", "").strip()
        if decision == "yes":
            row.update(
                {
                    "mapping_status": "matched",
                    "doc_id": proposed_doc_id,
                    "source": "tourapi",
                    "title": row.get("proposed_title"),
                    "description": row.get("proposed_description"),
                    "rag_image_path": row.get("proposed_image_path"),
                    "rag_image_url": row.get("proposed_image_url"),
                    "confidence": "verified",
                    "reason": "manual_review_approved",
                    "review_decision": "yes",
                    "review_note": note,
                }
            )
        else:
            row.update(
                {
                    "mapping_status": "unmatched",
                    "doc_id": None,
                    "source": None,
                    "title": None,
                    "description": None,
                    "rag_image_path": None,
                    "rag_image_url": None,
                    "confidence": "low",
                    "reason": (
                        "manual_review_rejected"
                        if decision == "no"
                        else "manual_review_pending"
                    ),
                    "review_decision": decision,
                    "review_note": note,
                }
            )
        final_rows.append(canonical_row(row))

    final_rows.sort(
        key=lambda row: (
            ("train", "validation", "test").index(str(row["split"])),
            str(row["image_key"]),
        )
    )
    status_counts = Counter(str(row["mapping_status"]) for row in final_rows)
    matched_rows = [row for row in final_rows if row["mapping_status"] == "matched"]
    unmatched_rows = [row for row in final_rows if row["mapping_status"] == "unmatched"]
    if any(row["doc_id"] is not None for row in unmatched_rows):
        raise RuntimeError("unmatched 행에 doc_id가 남아 있습니다.")
    if any(not row["description"] for row in matched_rows):
        raise RuntimeError("matched 행 중 description이 비어 있습니다.")

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "rag_mapping.jsonl"
    matched_path = output_dir / "rag_mapping_matched.jsonl"
    summary_path = output_dir / "rag_mapping_summary.json"
    atomic_write_jsonl(mapping_path, final_rows)
    atomic_write_jsonl(matched_path, matched_rows)
    summary = {
        "record_count": len(final_rows),
        "status_counts": dict(status_counts),
        "manual_review_count": len(expected_review_keys),
        "pending_review_count": len(pending_keys),
        "safety_checks": {
            "unique_image_key": len(final_rows) == len(set(row["image_key"] for row in final_rows)),
            "unmatched_doc_id_is_null": all(row["doc_id"] is None for row in unmatched_rows),
            "matched_description_is_present": all(bool(row["description"]) for row in matched_rows),
        },
        "outputs": {
            "mapping_db": str(mapping_path),
            "matched_only": str(matched_path),
        },
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
