#!/usr/bin/env python3
"""이미지 후보의 detailCommon2 설명을 일일 한도 내에서 수집한다.

호출 상태와 응답은 SQLite에 즉시 기록한다. 실행이 중단되어도 완료된
contentId는 다시 호출하지 않으며, 날짜별 호출 수를 보수적으로 계산해
설정한 일일 예산에 도달하면 자동 종료한다.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests

from test_connection import (
    BASE_URL,
    NORMAL_RESULT_CODES,
    extract_response,
    load_service_key,
    normalize_items,
    safe_json_response,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "targets" / "broad_image_candidates.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "details"
OPERATION = "detailCommon2"


class ApiResultError(RuntimeError):
    """정상 코드가 아닌 TourAPI 응답."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "광범위 대회 후보 JSONL의 contentId별 overview를 수집합니다. "
            "기본 일일 예산 900회에 도달하면 자동 종료합니다."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"이미지 후보 JSONL. 기본값: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 폴더. 기본값: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--daily-budget",
        type=int,
        default=900,
        help=(
            "이 스크립트가 하루 동안 허용할 detailCommon2 호출 수. "
            "다른 테스트 호출을 고려해 기본값은 900입니다."
        ),
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="이번 실행의 신규 호출 수 상한. 생략하면 남은 일일 예산을 사용합니다.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="요청별 제한 시간(초). 기본값: 60",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.1,
        help="호출 사이 대기 시간(초). 기본값: 0.1",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="API를 호출하지 않고 현재 캐시로 JSONL과 요약만 다시 생성합니다.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"JSONL {line_no}번째 줄이 올바르지 않습니다: {path}"
                    ) from exc
                if isinstance(value, dict):
                    rows.append(value)
    except OSError as exc:
        raise RuntimeError(f"입력 JSONL을 읽을 수 없습니다: {path}") from exc
    return rows


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS details (
            content_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            item_json TEXT,
            result_code TEXT,
            error_message TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            called_at TEXT NOT NULL,
            call_date TEXT NOT NULL,
            content_id TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def calls_on_date(connection: sqlite3.Connection, date_text: str) -> int:
    row = connection.execute(
        "SELECT COUNT(*) FROM call_log WHERE call_date = ?",
        (date_text,),
    ).fetchone()
    return int(row[0] if row else 0)


def processed_ids(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT content_id FROM details WHERE status IN ('success', 'empty')"
        )
    }


def begin_call(connection: sqlite3.Connection, content_id: str) -> int:
    now = datetime.now().astimezone()
    cursor = connection.execute(
        """
        INSERT INTO call_log(called_at, call_date, content_id, status)
        VALUES (?, ?, ?, 'started')
        """,
        (now.isoformat(timespec="seconds"), now.date().isoformat(), content_id),
    )
    connection.commit()
    return int(cursor.lastrowid)


def finish_call(
    connection: sqlite3.Connection,
    *,
    call_id: int,
    content_id: str,
    status: str,
    item: dict[str, Any] | None,
    result_code: str = "",
    error_message: str = "",
) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    connection.execute(
        "UPDATE call_log SET status = ? WHERE id = ?",
        (status, call_id),
    )
    connection.execute(
        """
        INSERT INTO details(
            content_id, status, item_json, result_code, error_message, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_id) DO UPDATE SET
            status = excluded.status,
            item_json = excluded.item_json,
            result_code = excluded.result_code,
            error_message = excluded.error_message,
            updated_at = excluded.updated_at
        """,
        (
            content_id,
            status,
            json.dumps(item, ensure_ascii=False) if item is not None else None,
            result_code,
            error_message[:500],
            now,
        ),
    )
    connection.commit()


def request_detail(
    service_key: str,
    content_id: str,
    timeout: float,
) -> tuple[dict[str, Any] | None, str, float]:
    url = f"{BASE_URL}/{OPERATION}"
    params = {
        "serviceKey": service_key,
        "MobileOS": "ETC",
        "MobileApp": "KCultureRAG",
        "_type": "json",
        "contentId": content_id,
    }
    started = time.perf_counter()
    response = requests.get(url, params=params, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = safe_json_response(response)
    if "response" not in data:
        top_level_code = str(data.get("resultCode", ""))
        top_level_message = str(data.get("resultMsg", ""))
        raise ApiResultError(
            f"code={top_level_code}, msg={top_level_message}"
        )
    header, body = extract_response(data)
    result_code = str(header.get("resultCode", ""))
    result_message = str(header.get("resultMsg", ""))
    if result_code not in NORMAL_RESULT_CODES:
        raise ApiResultError(f"code={result_code}, msg={result_message}")
    items = normalize_items(body.get("items"))
    return (items[0] if items else None), result_code, elapsed


def safe_error_message(exc: Exception) -> str:
    """요청 URL의 serviceKey가 로그나 SQLite에 남지 않도록 요약한다."""
    if isinstance(exc, requests.RequestException):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        status_text = f", HTTP {status_code}" if status_code is not None else ""
        return f"{type(exc).__name__}{status_text}"
    return f"{type(exc).__name__}: {exc}"


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\t\r\f\v ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
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


def build_export_record(
    candidate: dict[str, Any],
    detail_item: dict[str, Any],
) -> dict[str, Any]:
    title = clean_text(candidate.get("title"))
    content_id = str(candidate.get("contentid", "")).strip()
    return {
        "doc_id": content_id,
        "source": "tourapi",
        "title": title,
        "search_terms": [title] if title else [],
        "description": clean_text(detail_item.get("overview")),
        "image_path": "",
        "image_url": str(candidate.get("firstimage", "")).strip(),
        "thumbnail_url": str(candidate.get("firstimage2", "")).strip(),
        "content_type_id": str(candidate.get("contenttypeid", "")).strip(),
        "category_codes": [
            str(candidate.get(key, "")).strip()
            for key in ("cat1", "cat2", "cat3")
            if str(candidate.get(key, "")).strip()
        ],
        "copyright_type": str(candidate.get("cpyrhtDivCd", "")).strip(),
    }


def export_current(
    *,
    connection: sqlite3.Connection,
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    cached: dict[str, tuple[str, dict[str, Any] | None]] = {}
    for content_id, status, item_json in connection.execute(
        "SELECT content_id, status, item_json FROM details"
    ):
        item: dict[str, Any] | None = None
        if item_json:
            decoded = json.loads(item_json)
            if isinstance(decoded, dict):
                item = decoded
        cached[str(content_id)] = (str(status), item)

    records: list[dict[str, Any]] = []
    completed_count = 0
    empty_count = 0
    description_count = 0
    for candidate in candidates:
        content_id = str(candidate.get("contentid", "")).strip()
        cached_value = cached.get(content_id)
        if cached_value is None:
            continue
        status, item = cached_value
        if status == "empty":
            empty_count += 1
            completed_count += 1
            continue
        if status != "success" or item is None:
            continue
        completed_count += 1
        record = build_export_record(candidate, item)
        if record["description"]:
            description_count += 1
        records.append(record)

    records_path = output_dir / "tourapi_records_partial.jsonl"
    exported_count = atomic_write_jsonl(records_path, records)
    error_count_row = connection.execute(
        "SELECT COUNT(*) FROM details WHERE status = 'error'"
    ).fetchone()
    summary = {
        "endpoint": OPERATION,
        "candidate_count": len(candidates),
        "completed_count": completed_count,
        "remaining_count": len(candidates) - completed_count,
        "exported_record_count": exported_count,
        "nonempty_description_count": description_count,
        "empty_response_count": empty_count,
        "error_count": int(error_count_row[0] if error_count_row else 0),
        "records_path": str(records_path.resolve()),
    }
    atomic_write_json(output_dir / "detail_collection_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.daily_budget < 1:
        print("오류: --daily-budget은 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    if args.max_new is not None and args.max_new < 1:
        print("오류: --max-new는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    if args.timeout <= 0 or args.request_interval < 0:
        print("오류: timeout은 양수, request-interval은 0 이상이어야 합니다.", file=sys.stderr)
        return 2

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection: sqlite3.Connection | None = None
    try:
        candidates = load_jsonl(input_path)
        connection = open_database(output_dir / "detail_cache.sqlite3")
        today = datetime.now().astimezone().date().isoformat()
        used_today = calls_on_date(connection, today)
        remaining_budget = max(0, args.daily_budget - used_today)
        run_limit = remaining_budget
        if args.max_new is not None:
            run_limit = min(run_limit, args.max_new)

        completed = processed_ids(connection)
        pending = [
            row
            for row in candidates
            if str(row.get("contentid", "")).strip() not in completed
        ]
        target_completed_count = len(candidates) - len(pending)
        print(f"전체 이미지 후보       : {len(candidates):,}건")
        print(f"후보 중 기수집 완료    : {target_completed_count:,}건")
        print(f"오늘 기록된 상세 호출  : {used_today:,}회")
        print(f"이번 실행 최대 신규 호출: {run_limit:,}회")

        if not args.export_only and run_limit > 0 and pending:
            service_key = load_service_key()
            for index, candidate in enumerate(pending[:run_limit], start=1):
                content_id = str(candidate.get("contentid", "")).strip()
                call_id = begin_call(connection, content_id)
                try:
                    item, result_code, elapsed = request_detail(
                        service_key,
                        content_id,
                        args.timeout,
                    )
                except requests.RequestException as exc:
                    finish_call(
                        connection,
                        call_id=call_id,
                        content_id=content_id,
                        status="error",
                        item=None,
                        error_message=safe_error_message(exc),
                    )
                    status_code = getattr(
                        getattr(exc, "response", None), "status_code", None
                    )
                    print(
                        f"요청 실패: contentId={content_id}, {type(exc).__name__}. "
                        "완료되지 않은 항목은 다음 실행에서 재시도됩니다.",
                        file=sys.stderr,
                    )
                    if status_code in {401, 403, 429}:
                        print(
                            "인증 또는 호출 제한 가능성이 있어 수집을 중단합니다.",
                            file=sys.stderr,
                        )
                        break
                    continue
                except (ApiResultError, RuntimeError) as exc:
                    finish_call(
                        connection,
                        call_id=call_id,
                        content_id=content_id,
                        status="error",
                        item=None,
                        error_message=safe_error_message(exc),
                    )
                    print(
                        f"API 응답 오류로 중단: contentId={content_id}, "
                        f"{safe_error_message(exc)}. 현재 캐시는 보존됩니다.",
                        file=sys.stderr,
                    )
                    break

                status = "success" if item is not None else "empty"
                finish_call(
                    connection,
                    call_id=call_id,
                    content_id=content_id,
                    status=status,
                    item=item,
                    result_code=result_code,
                )
                if index == 1 or index % 50 == 0 or index == run_limit:
                    print(
                        f"진행: {index:,}/{run_limit:,} 신규 호출 "
                        f"(최근 {elapsed:.2f}초, {status})"
                    )
                if args.request_interval:
                    time.sleep(args.request_interval)

        summary = export_current(
            connection=connection,
            candidates=candidates,
            output_dir=output_dir,
        )
        print("=" * 60)
        print(f"상세 수집 완료        : {summary['completed_count']:,}건")
        print(f"남은 항목             : {summary['remaining_count']:,}건")
        print(f"설명 보유             : {summary['nonempty_description_count']:,}건")
        print(f"현재 JSONL            : {summary['records_path']}")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"상세 수집 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
