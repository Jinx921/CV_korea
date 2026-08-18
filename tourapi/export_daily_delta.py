#!/usr/bin/env python3
"""지정 날짜에 새로 성공한 detailCommon2 레코드만 JSONL로 내보낸다."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from collect_details import build_export_record, load_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = SCRIPT_DIR / "data" / "area_based" / "image_candidates.jsonl"
DEFAULT_DATABASE = SCRIPT_DIR / "data" / "details" / "detail_cache.sqlite3"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "details" / "daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="오늘 또는 지정 날짜에 새로 수집된 성공 레코드를 내보냅니다."
    )
    parser.add_argument(
        "--date",
        default=datetime.now().astimezone().date().isoformat(),
        help="내보낼 로컬 날짜(YYYY-MM-DD). 기본값은 오늘입니다.",
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    candidates = load_jsonl(args.candidates.expanduser().resolve())
    candidate_by_id = {
        str(row.get("contentid", "")).strip(): row
        for row in candidates
        if str(row.get("contentid", "")).strip()
    }

    connection = sqlite3.connect(args.database.expanduser().resolve())
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT d.content_id, d.item_json
            FROM call_log AS c
            JOIN details AS d ON d.content_id = c.content_id
            WHERE c.call_date = ?
              AND c.status = 'success'
              AND d.status = 'success'
            ORDER BY d.content_id
            """,
            (args.date,),
        ).fetchall()
    finally:
        connection.close()

    exported: list[dict[str, Any]] = []
    missing_candidate_count = 0
    for content_id, item_json in rows:
        candidate = candidate_by_id.get(str(content_id))
        if candidate is None:
            missing_candidate_count += 1
            continue
        item = json.loads(item_json)
        if not isinstance(item, dict):
            continue
        record = build_export_record(candidate, item)
        if record["description"]:
            exported.append(record)

    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / f"tourapi_collected_{args.date}.jsonl"
    summary_path = output_dir / f"tourapi_collected_{args.date}_summary.json"
    atomic_write_jsonl(output_path, exported)
    summary = {
        "date": args.date,
        "successful_call_content_count": len(rows),
        "exported_record_count": len(exported),
        "missing_candidate_count": missing_candidate_count,
        "output": str(output_path),
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
