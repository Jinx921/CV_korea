#!/usr/bin/env python3
"""대회 질문·선택지·OCR과 RAG 문서를 BM25로 매핑 후보화한다.

1단계는 제목·search_terms 인덱스로 넓은 후보 풀을 만들고, 설명 전체 인덱스의
후보도 합쳐 recall을 확보한다. 2단계는 question + options + OCR 전체 텍스트의
BM25 점수로 후보 풀을 재정렬한다. 점수가 0인 문서는 억지로 Top-K에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "processed"
DEFAULT_RAG_JSONL = (
    PROJECT_ROOT / "tourapi" / "data" / "details" / "tourapi_collected_records.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mapping" / "text_candidates"

TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+")

# 긴 조사부터 확인한다. 제거 후 한글 2음절 이상이 남을 때만 적용한다.
KOREAN_SUFFIXES = sorted(
    {
        "으로부터",
        "에서부터",
        "에게서는",
        "한테서는",
        "이라고는",
        "이라면서",
        "으로써",
        "으로서",
        "에서는",
        "에게서",
        "한테서",
        "이라고",
        "이라도",
        "까지는",
        "부터는",
        "으로는",
        "에서",
        "에게",
        "한테",
        "부터",
        "까지",
        "처럼",
        "보다",
        "이라고",
        "라는",
        "이며",
        "하고",
        "에도",
        "에서",
        "으로",
        "로부터",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "로",
        "와",
        "과",
        "도",
        "만",
        "께",
    },
    key=len,
    reverse=True,
)

# 문제 지시문은 모든 쿼리에 반복되지만 RAG 문서에서는 드물어 IDF가 비정상적으로
# 높아질 수 있으므로 검색 신호에서 제거한다.
QUIZ_STOPWORDS = {
    "사진",
    "이미지",
    "그림",
    "다음",
    "보기",
    "문항",
    "질문",
    "선택지",
    "설명",
    "내용",
    "관련",
    "해당",
    "대상",
    "장면",
    "모습",
    "가장",
    "모두",
    "각각",
    "무엇",
    "어느",
    "어떤",
    "옳은",
    "옳지",
    "알맞은",
    "적절한",
    "고르시오",
    "고르세요",
    "답하시오",
    "답하세요",
    "서술하시오",
    "서술하세요",
    "쓰시오",
    "작성하시오",
    "말하시오",
    "찾으시오",
    "관하여",
    "대하여",
    "바탕",
    "보고",
    "보이는",
    "보이지",
    "속의",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--rag-jsonl", type=Path, default=DEFAULT_RAG_JSONL)
    parser.add_argument(
        "--ocr-jsonl",
        type=Path,
        default=None,
        help="선택 사항: split, image_name, ocr_text가 있는 JSONL",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--routing-token-count", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
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


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).lower()


def strip_korean_suffix(token: str) -> str:
    if not re.fullmatch(r"[가-힣]+", token):
        return token
    for suffix in KOREAN_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if len(stem) >= 2:
                return stem
    return token


def tokenize(value: Any) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(normalize_text(value)):
        token = strip_korean_suffix(raw)
        if token in QUIZ_STOPWORDS:
            continue
        if re.fullmatch(r"[가-힣]+", token) and len(token) < 2:
            continue
        if re.fullmatch(r"[a-z]+", token) and len(token) < 2:
            continue
        tokens.append(token)
    return tokens


@dataclass
class BM25Index:
    documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.document_count = len(self.documents)
        if self.document_count == 0:
            raise ValueError("BM25 문서가 없습니다.")
        self.document_lengths = np.asarray(
            [len(document) for document in self.documents], dtype=np.float32
        )
        self.average_length = float(self.document_lengths.mean()) or 1.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_index, document in enumerate(self.documents):
            for token, frequency in Counter(document).items():
                postings[token].append((document_index, frequency))
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for token, values in postings.items():
            document_indices = np.asarray([value[0] for value in values], dtype=np.int32)
            frequencies = np.asarray([value[1] for value in values], dtype=np.float32)
            document_frequency = len(values)
            self.postings[token] = (document_indices, frequencies)
            self.idf[token] = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.document_count, dtype=np.float32)
        for token, query_frequency in Counter(query_tokens).items():
            posting = self.postings.get(token)
            if posting is None:
                continue
            document_indices, term_frequencies = posting
            length_normalizer = 1.0 - self.b + self.b * (
                self.document_lengths[document_indices] / self.average_length
            )
            denominator = term_frequencies + self.k1 * length_normalizer
            query_weight = 1.0 + math.log(float(query_frequency))
            scores[document_indices] += (
                self.idf[token]
                * ((term_frequencies * (self.k1 + 1.0)) / denominator)
                * query_weight
            )
        return scores


def positive_top_indices(scores: np.ndarray, count: int) -> list[int]:
    positive = np.flatnonzero(scores > 0)
    if not len(positive):
        return []
    count = min(int(count), len(positive))
    if count == len(positive):
        return positive[np.argsort(-scores[positive])].tolist()
    selected_positions = np.argpartition(-scores[positive], count - 1)[:count]
    selected = positive[selected_positions]
    return selected[np.argsort(-scores[selected])].tolist()


def build_query_text(record: dict[str, Any], ocr_text: str) -> str:
    parts = [str(record.get("question", ""))]
    parts.extend(str(option) for option in record.get("options", []) or [])
    if ocr_text:
        parts.append(ocr_text)
    return "\n".join(parts)


def normalized_compact(value: Any) -> str:
    return "".join(re.findall(r"[가-힣a-z0-9]+", normalize_text(value)))


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if not len(array):
        return {}
    return {
        str(level): round(float(np.quantile(array, level)), 6)
        for level in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    }


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    rag_path = args.rag_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    dataset_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        split_rows = load_jsonl(path)
        for row in split_rows:
            normalized = dict(row)
            normalized["split"] = split
            dataset_rows.append(normalized)

    rag_rows = load_jsonl(rag_path)
    unique_rag_rows: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for row in rag_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        description = str(row.get("description", "")).strip()
        if not doc_id or not description or doc_id in seen_doc_ids:
            continue
        normalized = dict(row)
        normalized["doc_id"] = doc_id
        unique_rag_rows.append(normalized)
        seen_doc_ids.add(doc_id)
    rag_rows = unique_rag_rows

    ocr_by_image: dict[tuple[str, str], str] = {}
    if args.ocr_jsonl is not None:
        for row in load_jsonl(args.ocr_jsonl.expanduser().resolve()):
            key = (str(row.get("split", "")), str(row.get("image_name", "")))
            ocr_by_image[key] = str(row.get("ocr_text", "")).strip()

    route_documents: list[list[str]] = []
    full_documents: list[list[str]] = []
    full_document_token_sets: list[set[str]] = []
    for row in rag_rows:
        title = str(row.get("title", ""))
        search_terms = " ".join(
            str(value) for value in row.get("search_terms", []) or []
        )
        description = str(row.get("description", ""))
        route_tokens = tokenize(f"{title} {title} {search_terms}")
        full_tokens = tokenize(
            f"{title} {title} {title} {search_terms} {search_terms} {description}"
        )
        route_documents.append(route_tokens)
        full_documents.append(full_tokens)
        full_document_token_sets.append(set(full_tokens))

    route_index = BM25Index(route_documents, k1=args.k1, b=args.b)
    full_index = BM25Index(full_documents, k1=args.k1, b=args.b)

    result_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for position, record in enumerate(dataset_rows, start=1):
        split = str(record["split"])
        image_name = str(record.get("image_name", ""))
        image_key = f"{split}:{image_name}"
        ocr_text = ocr_by_image.get((split, image_name), "")
        query_text = build_query_text(record, ocr_text)
        query_tokens = tokenize(query_text)

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        routing_tokens = sorted(
            (token for token in unique_query_tokens if token in route_index.idf),
            key=lambda token: route_index.idf[token],
            reverse=True,
        )[: args.routing_token_count]

        route_scores = route_index.score(routing_tokens)
        full_scores = full_index.score(query_tokens)
        route_top = positive_top_indices(route_scores, args.candidate_pool_size)
        broad_full_top = positive_top_indices(full_scores, args.candidate_pool_size)
        candidate_pool = sorted(set(route_top) | set(broad_full_top))
        candidate_pool.sort(key=lambda index: float(full_scores[index]), reverse=True)

        route_rank = {index: rank + 1 for rank, index in enumerate(route_top)}
        candidates: list[dict[str, Any]] = []
        compact_query = normalized_compact(query_text)
        for document_index in candidate_pool:
            if len(candidates) >= args.top_k:
                break
            score = float(full_scores[document_index])
            if score <= 0:
                continue
            document = rag_rows[document_index]
            title = str(document.get("title", ""))
            overlap = sorted(
                set(query_tokens) & full_document_token_sets[document_index],
                key=lambda token: full_index.idf.get(token, 0.0),
                reverse=True,
            )
            candidate = {
                "doc_id": str(document["doc_id"]),
                "source": str(document.get("source", "tourapi")),
                "title": title,
                "description": str(document.get("description", "")),
                "image_path": str(document.get("image_path", "")),
                "image_url": str(document.get("image_url", "")),
                "bm25_score": round(score, 6),
                "stage1_route_score": round(float(route_scores[document_index]), 6),
                "stage1_route_rank": route_rank.get(document_index),
                "matched_tokens": overlap[:20],
                "exact_title_in_query": bool(
                    normalized_compact(title)
                    and normalized_compact(title) in compact_query
                ),
            }
            candidates.append(candidate)

        top_score = candidates[0]["bm25_score"] if candidates else None
        second_score = candidates[1]["bm25_score"] if len(candidates) > 1 else None
        margin = (
            round(top_score - second_score, 6)
            if top_score is not None and second_score is not None
            else None
        )
        ratio = (
            round(top_score / second_score, 6)
            if top_score is not None and second_score not in (None, 0)
            else None
        )
        result = {
            "split": split,
            "question_id": str(record.get("question_id", "")),
            "image_name": image_name,
            "image_key": image_key,
            "question_form": str(record.get("question_form", "")),
            "query_token_count": len(query_tokens),
            "routing_tokens": routing_tokens,
            "ocr_used": bool(ocr_text),
            "top_score": top_score,
            "top1_top2_margin": margin,
            "top1_top2_ratio": ratio,
            "no_signal": not candidates,
            "candidates": candidates,
        }
        result_rows.append(result)
        for rank, candidate in enumerate(candidates, start=1):
            flat_rows.append(
                {
                    "split": split,
                    "question_id": result["question_id"],
                    "image_name": image_name,
                    "image_key": image_key,
                    "text_rank": rank,
                    **candidate,
                }
            )
        if position % 250 == 0:
            print(f"텍스트 검색 {position:,}/{len(dataset_rows):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "text_candidates.jsonl"
    pairs_path = output_dir / "text_candidate_pairs.jsonl"
    summary_path = output_dir / "text_candidate_summary.json"
    atomic_write_jsonl(candidates_path, result_rows)
    atomic_write_jsonl(pairs_path, flat_rows)

    top_scores = [row["top_score"] for row in result_rows if row["top_score"] is not None]
    margins = [
        row["top1_top2_margin"]
        for row in result_rows
        if row["top1_top2_margin"] is not None
    ]
    summary = {
        "method": "two_stage_bm25_question_options_ocr",
        "dataset_record_count": len(dataset_rows),
        "rag_record_count": len(rag_rows),
        "candidate_pair_count": len(flat_rows),
        "split_counts": {
            split: sum(row["split"] == split for row in result_rows)
            for split in ("train", "validation", "test")
        },
        "ocr_record_count": len(ocr_by_image),
        "ocr_used_query_count": sum(row["ocr_used"] for row in result_rows),
        "no_signal_query_count": sum(row["no_signal"] for row in result_rows),
        "top_score_quantiles": quantiles(top_scores),
        "top1_top2_margin_quantiles": quantiles(margins),
        "settings": {
            "candidate_pool_size": args.candidate_pool_size,
            "routing_token_count": args.routing_token_count,
            "top_k": args.top_k,
            "k1": args.k1,
            "b": args.b,
        },
        "note": "BM25 원점수는 쿼리 간 직접 임계값 비교에 사용하지 않습니다.",
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("저장:", candidates_path)
    print("저장:", pairs_path)
    print("저장:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
