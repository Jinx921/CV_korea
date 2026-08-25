from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_submission(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"제출 파일은 JSON 배열이어야 합니다: {path}")
    return data


def question_id(record: dict[str, Any]) -> str:
    return str(record.get("metadata", {}).get("question_id", ""))


def question_form(record: dict[str, Any]) -> str:
    return str(record.get("metadata", {}).get("question_form", "")).upper()


def answer(record: dict[str, Any]) -> str:
    return str(record.get("model_output", {}).get("answer", "")).strip()


def index_by_id(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = question_id(record)
        if not qid:
            raise ValueError(f"{label}에 question_id가 없는 문항이 있습니다.")
        if qid in result:
            raise ValueError(f"{label}에 중복 question_id가 있습니다: {qid}")
        result[qid] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="새 모델의 MC·SA 답변과 기존 모델의 LA 답변을 결합합니다."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_submission(args.baseline)
    champion = load_submission(args.champion)
    baseline_by_id = index_by_id(baseline, "baseline")
    champion_by_id = index_by_id(champion, "champion")

    if set(baseline_by_id) != set(champion_by_id):
        missing_in_baseline = sorted(set(champion_by_id) - set(baseline_by_id))
        missing_in_champion = sorted(set(baseline_by_id) - set(champion_by_id))
        raise ValueError(
            "두 제출 파일의 question_id 집합이 다릅니다. "
            f"baseline 누락={missing_in_baseline[:10]}, "
            f"champion 누락={missing_in_champion[:10]}"
        )

    output: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    form_counts: Counter[str] = Counter()
    changed_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()

    for champion_record in champion:
        qid = question_id(champion_record)
        baseline_record = baseline_by_id[qid]
        form = question_form(champion_record)

        if question_form(baseline_record) != form:
            raise ValueError(f"문항 유형이 서로 다릅니다: {qid}")
        if baseline_record.get("model_input") != champion_record.get("model_input"):
            raise ValueError(f"model_input이 서로 다릅니다: {qid}")

        if form == "LA":
            selected = copy.deepcopy(baseline_record)
            source = "baseline"
        elif form in {"MC", "SA"}:
            selected = copy.deepcopy(champion_record)
            source = "champion"
        else:
            raise ValueError(f"지원하지 않는 question_form입니다: {qid} / {form}")

        output.append(selected)
        source_counts[source] += 1
        form_counts[form] += 1
        if answer(baseline_record) != answer(champion_record):
            changed_counts[form] += 1
        if not answer(selected):
            empty_counts[form] += 1

    if len(output) != 800:
        raise ValueError(f"최종 제출 문항 수가 800개가 아닙니다: {len(output)}")
    if sum(empty_counts.values()):
        raise ValueError(f"빈 답변이 있습니다: {dict(empty_counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    summary = {
        "strategy": {
            "MC": "champion",
            "SA": "champion",
            "LA": "baseline",
        },
        "record_count": len(output),
        "form_counts": dict(sorted(form_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "answer_difference_between_inputs": dict(sorted(changed_counts.items())),
        "empty_answer_counts": dict(sorted(empty_counts.items())),
        "baseline": str(args.baseline),
        "champion": str(args.champion),
        "output": str(args.output),
    }
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
