#!/usr/bin/env python3
"""TourAPI areaBasedList2 전체 목록 수집 및 이미지 후보 선별.

각 페이지를 별도 JSON 파일로 원자적으로 저장하므로 실행이 중단되어도
이미 완료된 페이지는 다시 호출하지 않는다. 전체 페이지 수집이 끝나면
전체 JSONL과 대표 이미지가 있는 항목만 모은 JSONL을 생성한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from test_connection import (
    NORMAL_RESULT_CODES,
    extract_response,
    load_service_key,
    normalize_items,
    request_once,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "area_based"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "areaBasedList2 전체 목록을 페이지별로 저장하고 대표 이미지가 "
            "있는 항목을 선별합니다. 저장된 페이지는 재호출하지 않습니다."
        )
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="페이지당 항목 수. 기본값: 10000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP 요청 제한 시간(초). 기본값: 180",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.3,
        help="신규 API 호출 사이 대기 시간(초). 기본값: 0.3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 폴더. 기본값: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


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


def page_path(pages_dir: Path, page_no: int) -> Path:
    return pages_dir / f"page_{page_no:05d}.json"


def load_page(path: Path, expected_page_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"저장 페이지를 읽을 수 없습니다: {path}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RuntimeError(f"저장 페이지 형식이 올바르지 않습니다: {path}")

    saved_page_size = int(payload.get("requested_page_size") or 0)
    if saved_page_size != expected_page_size:
        raise RuntimeError(
            f"기존 페이지 크기({saved_page_size})와 현재 --page-size"
            f"({expected_page_size})가 다릅니다. 동일한 값을 사용하거나 "
            "새 --output-dir을 지정하세요."
        )
    return payload


def fetch_and_save_page(
    *,
    service_key: str,
    page_no: int,
    page_size: int,
    timeout: float,
    destination: Path,
) -> dict[str, Any]:
    data, elapsed = request_once(
        service_key=service_key,
        page_size=page_size,
        page_no=page_no,
        timeout=timeout,
    )
    header, body = extract_response(data)
    result_code = str(header.get("resultCode", ""))
    result_message = str(header.get("resultMsg", ""))
    if result_code not in NORMAL_RESULT_CODES:
        raise RuntimeError(
            f"API 오류: page={page_no}, code={result_code}, msg={result_message}"
        )

    items = normalize_items(body.get("items"))
    payload = {
        "page_no": int(body.get("pageNo") or page_no),
        "requested_page_size": page_size,
        "response_page_size": int(body.get("numOfRows") or 0),
        "total_count": int(body.get("totalCount") or 0),
        "item_count": len(items),
        "elapsed_seconds": round(elapsed, 3),
        "items": items,
    }
    atomic_write_json(destination, payload)
    return payload


def candidate_record(row: dict[str, Any]) -> dict[str, Any]:
    """detailCommon2와 이미지 다운로드 단계에 필요한 필드만 남긴다."""
    selected_fields = (
        "contentid",
        "contenttypeid",
        "title",
        "firstimage",
        "firstimage2",
        "cat1",
        "cat2",
        "cat3",
        "lclsSystm1",
        "lclsSystm2",
        "lclsSystm3",
        "cpyrhtDivCd",
        "createdtime",
        "modifiedtime",
    )
    return {
        field: row.get(field, "")
        for field in selected_fields
        if field in row
    }


def build_outputs(
    *,
    pages_dir: Path,
    output_dir: Path,
    total_pages: int,
    page_size: int,
    api_total_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for page_no in range(1, total_pages + 1):
        payload = load_page(page_path(pages_dir, page_no), page_size)
        rows.extend(
            item for item in payload["items"] if isinstance(item, dict)
        )

    unique_rows: list[dict[str, Any]] = []
    seen_content_ids: set[str] = set()
    missing_content_id_count = 0
    duplicate_content_id_count = 0
    for row in rows:
        content_id = str(row.get("contentid", "")).strip()
        if not content_id:
            missing_content_id_count += 1
            continue
        if content_id in seen_content_ids:
            duplicate_content_id_count += 1
            continue
        seen_content_ids.add(content_id)
        unique_rows.append(row)

    image_candidates = [
        candidate_record(row)
        for row in unique_rows
        if str(row.get("firstimage", "")).strip()
    ]

    all_path = output_dir / "area_based_all.jsonl"
    candidates_path = output_dir / "image_candidates.jsonl"
    all_count = atomic_write_jsonl(all_path, unique_rows)
    candidate_count = atomic_write_jsonl(candidates_path, image_candidates)

    summary = {
        "endpoint": "areaBasedList2",
        "page_size": page_size,
        "page_count": total_pages,
        "api_total_count": api_total_count,
        "downloaded_row_count": len(rows),
        "unique_content_count": all_count,
        "image_candidate_count": candidate_count,
        "without_representative_image_count": all_count - candidate_count,
        "missing_content_id_count": missing_content_id_count,
        "duplicate_content_id_count": duplicate_content_id_count,
        "outputs": {
            "all": str(all_path.resolve()),
            "image_candidates": str(candidates_path.resolve()),
        },
    }
    atomic_write_json(output_dir / "collection_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.page_size < 1:
        print("오류: --page-size는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("오류: --timeout은 0보다 커야 합니다.", file=sys.stderr)
        return 2
    if args.request_interval < 0:
        print("오류: --request-interval은 0 이상이어야 합니다.", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    try:
        service_key = load_service_key()

        first_path = page_path(pages_dir, 1)
        if first_path.exists():
            first_page = load_page(first_path, args.page_size)
            print("1페이지: 저장된 파일을 재사용합니다.")
        else:
            first_page = fetch_and_save_page(
                service_key=service_key,
                page_no=1,
                page_size=args.page_size,
                timeout=args.timeout,
                destination=first_path,
            )
            print(
                f"1페이지: {first_page['item_count']:,}건 저장 "
                f"({first_page['elapsed_seconds']:.2f}초)"
            )

        total_count = int(first_page.get("total_count") or 0)
        if total_count < 1:
            raise RuntimeError("API의 totalCount가 0입니다.")
        total_pages = math.ceil(total_count / args.page_size)
        print(f"전체 {total_count:,}건, 총 {total_pages:,}페이지")

        for page_no in range(2, total_pages + 1):
            destination = page_path(pages_dir, page_no)
            if destination.exists():
                cached = load_page(destination, args.page_size)
                print(
                    f"{page_no}페이지: 저장된 {cached['item_count']:,}건 재사용"
                )
                continue

            if args.request_interval:
                time.sleep(args.request_interval)
            payload = fetch_and_save_page(
                service_key=service_key,
                page_no=page_no,
                page_size=args.page_size,
                timeout=args.timeout,
                destination=destination,
            )
            print(
                f"{page_no}페이지: {payload['item_count']:,}건 저장 "
                f"({payload['elapsed_seconds']:.2f}초)"
            )

        summary = build_outputs(
            pages_dir=pages_dir,
            output_dir=output_dir,
            total_pages=total_pages,
            page_size=args.page_size,
            api_total_count=total_count,
        )
        print("=" * 60)
        print(f"전체 고유 항목       : {summary['unique_content_count']:,}건")
        print(f"대표 이미지 보유 항목: {summary['image_candidate_count']:,}건")
        print(
            "후보 JSONL           : "
            f"{summary['outputs']['image_candidates']}"
        )
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        status_text = f", HTTP {status_code}" if status_code is not None else ""
        print(
            "수집 실패: HTTP 요청 오류"
            f"({type(exc).__name__}{status_text}). 저장된 페이지까지는 유지되며 "
            "같은 명령으로 다시 실행하면 이어서 수집합니다.",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"수집 실패: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
