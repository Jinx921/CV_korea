#!/usr/bin/env python3
"""한국관광공사 TourAPI 연결 및 페이지 크기 테스트.

이 스크립트는 실행할 때마다 ``areaBasedList2``를 정확히 한 번 호출한다.
API 키는 ``TOUR_API_KEY`` 환경변수 또는 같은 폴더의 ``.env``에서 읽으며,
화면이나 파일에 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

try:
    import requests
except ImportError as exc:  # pragma: no cover - 설치 안내용
    raise SystemExit(
        "requests가 설치되어 있지 않습니다. "
        "`python -m pip install requests python-dotenv`를 실행하세요."
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover - 설치 안내용
    raise SystemExit(
        "python-dotenv가 설치되어 있지 않습니다. "
        "`python -m pip install requests python-dotenv`를 실행하세요."
    ) from exc


BASE_URL = "https://apis.data.go.kr/B551011/KorService2"
OPERATION = "areaBasedList2"
NORMAL_RESULT_CODES = {"00", "0000"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TourAPI 연결을 확인하고 totalCount와 실제 반환 건수를 출력합니다. "
            "실행당 API 호출은 1회입니다."
        )
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1,
        help="한 번에 요청할 항목 수. 기본값은 1입니다.",
    )
    parser.add_argument(
        "--page-no",
        type=int,
        default=1,
        help="요청할 페이지 번호. 기본값은 1입니다.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP 요청 제한 시간(초). 기본값은 60초입니다.",
    )
    return parser.parse_args()


def load_service_key() -> str:
    # 이 저장소에서 기존에 사용하던 프로젝트 루트 .env를 재사용한다.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    raw_key = os.getenv("TOUR_API_KEY", "").strip()
    if not raw_key:
        raise RuntimeError(
            "TOUR_API_KEY가 없습니다. "
            "프로젝트 루트 .env에 `TOUR_API_KEY=...`를 추가하세요."
        )

    # 포털에서 받은 Encoding 키(%2F, %2B, %3D 포함)와 Decoding 키를 모두 지원한다.
    return unquote(raw_key)


def normalize_items(raw_items: Any) -> list[dict[str, Any]]:
    if raw_items in (None, ""):
        return []
    if isinstance(raw_items, dict):
        item = raw_items.get("item", [])
    else:
        item = raw_items

    if item in (None, ""):
        return []
    if isinstance(item, dict):
        return [item]
    if isinstance(item, list):
        return [row for row in item if isinstance(row, dict)]
    return []


def safe_json_response(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except requests.JSONDecodeError as exc:
        preview = response.text[:500].replace("\n", " ")
        raise RuntimeError(
            "JSON 응답을 받지 못했습니다. "
            f"HTTP {response.status_code}, 응답 앞부분: {preview}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("API 응답의 최상위 구조가 JSON 객체가 아닙니다.")
    return data


def request_once(
    service_key: str,
    page_size: int,
    page_no: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    url = f"{BASE_URL}/{OPERATION}"
    params = {
        "serviceKey": service_key,
        "pageNo": page_no,
        "numOfRows": page_size,
        "MobileOS": "ETC",
        "MobileApp": "KCultureRAG",
        "_type": "json",
    }

    started_at = time.perf_counter()
    response = requests.get(url, params=params, timeout=timeout)
    elapsed = time.perf_counter() - started_at
    response.raise_for_status()
    return safe_json_response(response), elapsed


def extract_response(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = data.get("response")
    if not isinstance(root, dict):
        raise RuntimeError(
            "응답에 `response` 객체가 없습니다. "
            f"확인된 최상위 필드: {sorted(data.keys())}"
        )

    header = root.get("header")
    body = root.get("body")
    if not isinstance(header, dict) or not isinstance(body, dict):
        raise RuntimeError("응답에 `header` 또는 `body` 객체가 없습니다.")
    return header, body


def print_report(
    header: dict[str, Any],
    body: dict[str, Any],
    requested_page_size: int,
    elapsed: float,
) -> None:
    result_code = str(header.get("resultCode", ""))
    result_message = str(header.get("resultMsg", ""))
    total_count = int(body.get("totalCount") or 0)
    response_page_size = int(body.get("numOfRows") or 0)
    page_no = int(body.get("pageNo") or 0)
    items = normalize_items(body.get("items"))

    print("=" * 60)
    print("TourAPI 연결 테스트 결과")
    print("=" * 60)
    print(f"Endpoint             : {BASE_URL}/{OPERATION}")
    print(f"결과 코드            : {result_code}")
    print(f"결과 메시지          : {result_message}")
    print(f"전체 데이터 수       : {total_count:,}")
    print(f"요청 pageNo          : {page_no}")
    print(f"요청 numOfRows       : {requested_page_size:,}")
    print(f"응답 numOfRows       : {response_page_size:,}")
    print(f"실제 반환 항목 수    : {len(items):,}")
    print(f"응답 시간            : {elapsed:.2f}초")

    if total_count > 0 and requested_page_size > 0:
        expected_calls = math.ceil(total_count / requested_page_size)
        print(f"현재 page-size 기준 예상 호출 수: {expected_calls:,}회")

    if items:
        first_item = items[0]
        print("첫 번째 항목 필드:")
        print("  " + ", ".join(sorted(first_item.keys())))

        safe_sample_fields = (
            "contentid",
            "contenttypeid",
            "title",
            "addr1",
            "cat1",
            "cat2",
            "cat3",
            "firstimage",
            "cpyrhtDivCd",
            "modifiedtime",
        )
        safe_sample = {
            key: first_item[key]
            for key in safe_sample_fields
            if key in first_item
        }
        print("첫 번째 항목 예시:")
        print(json.dumps(safe_sample, ensure_ascii=False, indent=2))

    if result_code not in NORMAL_RESULT_CODES:
        raise RuntimeError(
            f"API가 오류 코드를 반환했습니다: {result_code} {result_message}"
        )

    if requested_page_size >= 10_000:
        if len(items) == requested_page_size:
            print("판정: numOfRows=10000 요청이 정상적으로 적용됐습니다.")
        elif len(items) == total_count and total_count < requested_page_size:
            print("판정: 전체 데이터가 page-size보다 적어 모두 반환됐습니다.")
        else:
            print(
                "판정: 요청한 page-size보다 적게 반환됐습니다. "
                "응답 제한 또는 마지막 페이지 여부를 확인하세요."
            )


def main() -> int:
    args = parse_args()
    if args.page_size < 1:
        print("오류: --page-size는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    if args.page_no < 1:
        print("오류: --page-no는 1 이상이어야 합니다.", file=sys.stderr)
        return 2

    try:
        service_key = load_service_key()
        data, elapsed = request_once(
            service_key=service_key,
            page_size=args.page_size,
            page_no=args.page_no,
            timeout=args.timeout,
        )
        header, body = extract_response(data)
        print_report(
            header=header,
            body=body,
            requested_page_size=args.page_size,
            elapsed=elapsed,
        )
    except requests.RequestException as exc:
        # requests 예외 문자열에는 serviceKey가 포함된 전체 URL이 들어갈 수
        # 있으므로 예외 본문을 그대로 출력하지 않는다.
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        status_text = f", HTTP {status_code}" if status_code is not None else ""
        print(
            "연결 테스트 실패: HTTP 요청 오류"
            f"({type(exc).__name__}{status_text}). "
            "네트워크 연결, API 승인 상태, Endpoint를 확인하세요.",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"연결 테스트 실패: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
