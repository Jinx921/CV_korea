#!/usr/bin/env python3
"""Kanana validation_predictions.csv의 SA exact-match 오류를 진단한다."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = PROJECT_ROOT / "outputs" / "processed" / "validation.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--processed-jsonl", type=Path, default=DEFAULT_PROCESSED)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = re.sub(r"^\s*(?:정답은|정답|답은|답)\s*[:：]?\s*", "", text)
    text = re.sub(r"\s*(?:입니다|이다|이에요|예요)\s*[.!?。]?$", "", text)
    text = text.strip(" \t\r\n\"'“”‘’`.,!?。·:：;；()[]{}<>")
    return re.sub(r"\s+", " ", text)


def compact_text(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", normalized_text(value)).lower()


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def measured_length(text: str, unit: str | None) -> int | None:
    cleaned = normalized_text(text)
    if unit == "어절":
        return len([token for token in cleaned.split() if token])
    if unit in {"음절", "글자"}:
        return len(re.findall(r"[0-9A-Za-z가-힣]", cleaned))
    return None


def main() -> int:
    args = parse_args()
    prediction_path = args.predictions_csv.expanduser().resolve()
    processed_path = args.processed_jsonl.expanduser().resolve()

    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    metadata_rows = load_jsonl(processed_path)
    metadata_by_id = {str(row["question_id"]): row for row in metadata_rows}

    sa_rows = [row for row in predictions if row.get("question_form") == "SA"]
    if len(sa_rows) != 52:
        raise RuntimeError(f"예상 SA 52건과 다릅니다: {len(sa_rows)}")

    details = []
    categories = Counter()
    for row in sa_rows:
        question_id = str(row["question_id"])
        metadata = metadata_by_id[question_id]
        prediction = str(row.get("prediction", ""))
        raw = str(row.get("raw", prediction))
        target = str(row.get("target", metadata.get("answer", "")))
        exact = bool_value(row.get("exact_match"))
        prediction_compact = compact_text(prediction)
        raw_compact = compact_text(raw)
        target_compact = compact_text(target)
        distance = levenshtein(prediction_compact, target_compact)
        requested_length_raw = metadata.get("requested_length")
        requested_length = (
            int(requested_length_raw) if requested_length_raw is not None else None
        )
        requested_unit = metadata.get("requested_unit")
        actual_length = measured_length(prediction, requested_unit)
        length_ok = (
            None
            if requested_length is None or actual_length is None
            else actual_length == requested_length
        )

        if exact:
            category = "exact"
        elif prediction_compact == target_compact or raw_compact == target_compact:
            category = "format_recoverable"
        elif target_compact and (
            target_compact in raw_compact or raw_compact in target_compact
        ):
            category = "substring_or_extra_text"
        elif distance <= 1:
            category = "one_edit_near_miss"
        elif length_ok is False:
            category = "length_constraint_violation"
        else:
            category = "semantic_wrong"
        categories[category] += 1

        details.append(
            {
                "question_id": question_id,
                "ocr_cue": bool(metadata.get("ocr_cue")),
                "rag_applied": bool_value(row.get("rag_applied")),
                "exact": exact,
                "category": category,
                "prediction": prediction,
                "target": target,
                "edit_distance": distance,
                "requested_length": requested_length,
                "requested_unit": requested_unit,
                "actual_length": actual_length,
                "length_ok": length_ok,
                "question": metadata.get("question", ""),
            }
        )

    exact_count = sum(row["exact"] for row in details)
    ocr_rows = [row for row in details if row["ocr_cue"]]
    non_ocr_rows = [row for row in details if not row["ocr_cue"]]
    constrained = [row for row in details if row["length_ok"] is not None]
    wrong = [row for row in details if not row["exact"]]

    summary = {
        "sa_count": len(details),
        "exact_count": exact_count,
        "exact_rate": exact_count / len(details),
        "ocr_cue": {
            "count": len(ocr_rows),
            "exact_count": sum(row["exact"] for row in ocr_rows),
            "exact_rate": sum(row["exact"] for row in ocr_rows) / len(ocr_rows),
        },
        "non_ocr": {
            "count": len(non_ocr_rows),
            "exact_count": sum(row["exact"] for row in non_ocr_rows),
            "exact_rate": sum(row["exact"] for row in non_ocr_rows) / len(non_ocr_rows),
        },
        "length_constraint": {
            "count": len(constrained),
            "compliant_count": sum(row["length_ok"] is True for row in constrained),
            "compliance_rate": sum(row["length_ok"] is True for row in constrained)
            / len(constrained),
            "wrong_but_length_compliant": sum(
                not row["exact"] and row["length_ok"] is True for row in constrained
            ),
        },
        "error_categories": dict(categories),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n오답 상세")
    for row in wrong:
        print(
            f"{row['question_id']} | {row['category']} | OCR={row['ocr_cue']} | "
            f"길이={row['actual_length']}/{row['requested_length']} {row['requested_unit']} | "
            f"예측={row['prediction']!r} | 정답={row['target']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
