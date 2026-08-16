#!/usr/bin/env python3
"""대회 광범위 후보 외 TourAPI 항목을 카테고리별로 고르게 정렬한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ALL = SCRIPT_DIR / "data" / "area_based" / "image_candidates.jsonl"
DEFAULT_PRIMARY = SCRIPT_DIR / "data" / "targets" / "broad_image_candidates.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "targets" / "diverse_extra_candidates.jsonl"

# 한국문화 질의응답에 직접적인 유형을 먼저 순환하되 모든 유형을 포함한다.
CONTENT_TYPE_PRIORITY = {
    "14": 0,  # 문화시설
    "12": 1,  # 관광지
    "15": 2,  # 축제/공연/행사
    "25": 3,  # 여행코스
    "39": 4,  # 음식
    "38": 5,  # 쇼핑
    "28": 6,  # 레포츠
    "32": 7,  # 숙박
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="광범위 후보 외 이미지 항목을 콘텐츠 유형·카테고리 순환 순서로 만듭니다."
    )
    parser.add_argument("--all-input", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--primary-input", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


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


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    content_type = str(row.get("contenttypeid", "") or "unknown")
    category = str(row.get("cat2", "") or row.get("cat1", "") or "unknown")
    return content_type, category


def main() -> int:
    args = parse_args()
    all_rows = load_jsonl(args.all_input.expanduser().resolve())
    primary_rows = load_jsonl(args.primary_input.expanduser().resolve())
    primary_ids = {str(row.get("contentid", "")) for row in primary_rows}

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        content_id = str(row.get("contentid", ""))
        if not content_id or content_id in primary_ids:
            continue
        candidate = dict(row)
        candidate["selection_method"] = "diversity_round_robin"
        candidate["candidate_tier"] = "diverse_extra"
        candidate["diversity_group"] = ":".join(group_key(row))
        groups[group_key(row)].append(candidate)

    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                str(row.get("modifiedtime", "")),
                str(row.get("contentid", "")),
            ),
            reverse=True,
        )

    ordered_keys = sorted(
        groups,
        key=lambda key: (CONTENT_TYPE_PRIORITY.get(key[0], 99), key[0], key[1]),
    )
    queues = {key: deque(groups[key]) for key in ordered_keys}
    ordered: list[dict[str, Any]] = []
    while queues:
        for key in list(queues):
            queue = queues[key]
            ordered.append(queue.popleft())
            if not queue:
                del queues[key]

    output_path = args.output.expanduser().resolve()
    atomic_write_jsonl(output_path, ordered)
    type_counts = Counter(str(row.get("contenttypeid", "unknown")) for row in ordered)
    summary = {
        "all_image_candidate_count": len(all_rows),
        "excluded_primary_count": len(primary_ids),
        "diverse_extra_candidate_count": len(ordered),
        "group_count": len(groups),
        "content_type_counts": dict(sorted(type_counts.items())),
        "output": str(output_path),
    }
    atomic_write_json(output_path.with_name("diverse_extra_summary.json"), summary)
    print(f"전체 이미지 후보        : {len(all_rows):,}건")
    print(f"광범위 후보 제외        : {len(primary_ids):,}건")
    print(f"다양성 추가 후보        : {len(ordered):,}건")
    print(f"콘텐츠유형·카테고리 그룹: {len(groups):,}개")
    print(f"출력                    : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
