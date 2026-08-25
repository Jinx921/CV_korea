from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "kanana_unified_rag_champion_runtime.py"
OUTPUT_PATH = ROOT / "notebooks" / "kanana15v_3b_unified_rag_champion.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "accelerator": "GPU",
    "colab": {
        "name": OUTPUT_PATH.name,
        "provenance": [],
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.x"},
}

nb.cells = [
    md(
        """
        ## Kanana-1.5-V 3B Unified RAG Champion

        목표는 기존 최고점 **44.8395**를 넘어 47점 이상, 가능하면 50점 이상을
        탐색하는 것입니다. 점수 달성을 보장하는 노트북은 아니며, 검증 지표와
        리더보드 결과로 최종 판단해야 합니다.

        핵심 변경점:

        - 팀 AIVQA 파이프라인: Shared LoRA 2 epoch + MC/SA/LA 유형별 best Adapter
        - 2,500건 통합 DB의 KURE 텍스트 검색 + 가용할 때만 CLIP 이미지 검색
        - 실제 RAG 이미지가 없으면 README 규칙대로 text-only Qdrant로 자동 전환
        - 검색 결과를 강제로 넣지 않는 **고신뢰 selective RAG**
        - 유형별 RAG 설명 길이 제한: MC 700 / SA 550 / LA 1,200자
        - 유형별 학습 예산: MC 3 / SA 5 / LA 3 epoch, 매 epoch 검증 최고점 저장
        - SA는 자유 생성 후 길이 위반 시에만 음절·어절 hard retry
        - 검색 cache, Qdrant DB, Adapter, 제출 파일을 Drive에 저장

        Colab L4 또는 A100 GPU에서 위에서부터 실행하세요. 설치 셀 뒤 런타임이
        자동 재시작되면 다시 `런타임 > 모두 실행`을 누르세요.
        """
    ),
    md("### 0. 환경 설치"),
    code(
        r'''
        import os
        import subprocess
        import sys
        from pathlib import Path


        marker = Path("/content/.aivqa_unified_champion_env_v1")
        if not marker.exists():
            def pip_install(*packages):
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", *packages]
                )

            subprocess.call(
                [
                    sys.executable, "-m", "pip", "uninstall", "-q", "-y",
                    "gradio", "gradio_client", "torchao", "torchaudio",
                ]
            )
            pip_install(
                "transformers>=4.57.0,<5.0.0",
                "accelerate>=1.0.0,<2.0.0",
                "peft>=0.17.0,<1.0.0",
                "qdrant-client>=1.13.0,<2.0.0",
                "sentence-transformers>=5.0.0,<6.0.0",
                "bitsandbytes>=0.46.0,<1.0.0",
                "timm>=1.0.0,<2.0.0",
                "einops>=0.8.0,<1.0.0",
                "omegaconf>=2.3.0,<3.0.0",
                "safetensors>=0.4.3,<1.0.0",
                "pandas>=2.0.0,<3.0.0",
                "tqdm>=4.66.0,<5.0.0",
            )
            pip_install(
                "--no-cache-dir", "--force-reinstall", "Pillow>=12.0.0,<13.0.0"
            )
            marker.write_text("ready", encoding="utf-8")
            print("설치 완료. 런타임을 재시작합니다.")
            print("재연결 후 첫 셀부터 다시 모두 실행하세요.")
            os.kill(os.getpid(), 9)

        import importlib.metadata
        import PIL
        import transformers

        print("Python:", sys.version.split()[0])
        print("Pillow:", PIL.__version__)
        print("Transformers:", transformers.__version__)
        print("Qdrant client:", importlib.metadata.version("qdrant-client"))
        '''
    ),
    md("### 1. Google Drive 연결 및 실험 설정"),
    code(
        r'''
        from google.colab import drive
        drive.mount("/content/drive")
        '''
    ),
    code(
        r'''
        import argparse
        import copy
        import gc
        import hashlib
        import json
        import os
        import random
        import re
        import shutil
        import subprocess
        import sys
        import unicodedata
        import zipfile
        from collections import Counter
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import torch
        from PIL import Image
        from tqdm.auto import tqdm


        DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
        DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"
        RAG_BUNDLE_DIR = DRIVE_PROJECT_DIR / "unified_rag_db"
        RAG_JSONL_PATH = RAG_BUNDLE_DIR / "unified_rag.jsonl"

        REPO_URL = "https://github.com/kjh0902/AIVQA.git"
        REPO_COMMIT = "e961b1f3fbb5d77ef4414e7644e63b913177c827"
        MODEL_ID = "kakaocorp/kanana-1.5-v-3b-instruct"
        KANANA_REVISION = "2e00ef13ccec2e99459a8eada18a1bfd05bff44b"

        RUN_NAME = "kanana15v_3b_unified_rag_champion_v1"
        COLAB_ROOT = Path("/content/aivqa_unified_champion")
        REPO_DIR = COLAB_ROOT / "AIVQA"
        EXTRACT_DIR = COLAB_ROOT / "extracted"
        DATASET_ROOT = COLAB_ROOT / "dataset_root"
        LOCAL_RAG_DIR = COLAB_ROOT / "rag_bundle"
        LOCAL_RAG_JSONL = LOCAL_RAG_DIR / "unified_rag.jsonl"
        LOCAL_QDRANT_DIR = COLAB_ROOT / "qdrant_storage"
        MODEL_CACHE_DIR = COLAB_ROOT / "model_cache"
        RUN_DIR = DRIVE_PROJECT_DIR / "outputs" / RUN_NAME

        # Champion 전체 학습 노트북이므로 True로 유지합니다.
        DO_TRAIN = True
        REBUILD_QDRANT = False

        SHARED_EPOCHS = 2
        TYPE_EPOCHS_BY_FORM = {"MC": 3, "SA": 5, "LA": 3}
        SHARED_LEARNING_RATE = 5e-5
        TYPE_LEARNING_RATE = 2e-5
        GRADIENT_ACCUMULATION_STEPS = 8
        MAX_LENGTH = 4096
        MIN_PIXELS = 100 * 28 * 28
        MAX_PIXELS = 600 * 28 * 28
        SEED = 42

        # 점수가 서로 다른 KURE와 CLIP을 단순 합산하지 않습니다.
        RETRIEVAL_BASE_THRESHOLD = 0.80
        DUAL_TEXT_THRESHOLD = 0.82
        DUAL_IMAGE_THRESHOLD = 0.82
        STRONG_TEXT_THRESHOLD = 0.93
        STRONG_IMAGE_THRESHOLD = 0.965
        RAG_TOP_K = 1
        RAG_CONTEXT_LIMITS = {"MC": 700, "SA": 550, "LA": 1200}

        for path in (
            DRIVE_PROJECT_DIR, COLAB_ROOT, EXTRACT_DIR, DATASET_ROOT,
            LOCAL_RAG_DIR, MODEL_CACHE_DIR, RUN_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        print("대회 ZIP:", DRIVE_ZIP_DIR)
        print("통합 RAG:", RAG_JSONL_PATH)
        print("실험 결과:", RUN_DIR)
        print("학습:", SHARED_EPOCHS, "+", TYPE_EPOCHS_BY_FORM)
        '''
    ),
    code(
        r'''
        if not torch.cuda.is_available():
            raise RuntimeError("Colab 런타임 유형을 GPU로 변경하세요.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print("GPU:", gpu_name)
        print(f"VRAM: {gpu_mem_gb:.1f} GiB")
        print("학습 dtype:", dtype_name)
        if DO_TRAIN and gpu_mem_gb < 20:
            raise RuntimeError("일반 LoRA 학습은 L4/A100급(20GB 이상) GPU를 권장합니다.")
        '''
    ),
]

nb.cells.extend(
    [
        md("### 2. 팀 AIVQA 코드 고정 및 대회 데이터 준비"),
        code(
            r'''
            if not (REPO_DIR / ".git").is_dir():
                subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
            else:
                subprocess.check_call(["git", "-C", str(REPO_DIR), "fetch", "origin"])

            subprocess.check_call(
                ["git", "-C", str(REPO_DIR), "checkout", "--detach", REPO_COMMIT]
            )
            actual_commit = subprocess.check_output(
                ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
            ).strip()
            assert actual_commit == REPO_COMMIT, (actual_commit, REPO_COMMIT)
            if str(REPO_DIR) not in sys.path:
                sys.path.insert(0, str(REPO_DIR))
            print("AIVQA commit:", actual_commit)
            '''
        ),
        code(
            r'''
            EXPECTED_ARCHIVES = [
                "한국문화 멀티모달 질의응답.zip",
                "train.zip",
                "validation.zip",
                "test.zip",
            ]


            def normalized_name(text):
                return unicodedata.normalize("NFC", str(text))


            def resolve_archive(directory, expected_name):
                expected_nfc = normalized_name(expected_name)
                matches = [
                    path for path in directory.glob("*.zip")
                    if normalized_name(path.name) == expected_nfc
                ]
                if len(matches) != 1:
                    found = [path.name for path in directory.glob("*.zip")]
                    raise FileNotFoundError(
                        f"{expected_name!r}을 찾지 못했습니다. 현재 ZIP: {found}"
                    )
                return matches[0]


            archive_paths = [
                resolve_archive(DRIVE_ZIP_DIR, name) for name in EXPECTED_ARCHIVES
            ]
            archive_state = {
                path.name: {"size": path.stat().st_size, "mtime": path.stat().st_mtime_ns}
                for path in archive_paths
            }
            state_path = EXTRACT_DIR / ".archive_state.json"
            previous = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file() else None
            )
            if previous == archive_state:
                print("동일한 ZIP 압축 해제 결과를 재사용합니다.")
            else:
                for archive_path in archive_paths:
                    print("압축 해제:", archive_path.name)
                    with zipfile.ZipFile(archive_path) as archive:
                        archive.extractall(EXTRACT_DIR)
                state_path.write_text(
                    json.dumps(archive_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            '''
        ),
        code(
            r'''
            def find_split_json(split):
                matches = [
                    path for path in EXTRACT_DIR.rglob("*.json")
                    if normalized_name(path.stem).endswith(f"_{split}")
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"{split} JSON 후보가 1개가 아닙니다: {matches}")
                return matches[0]


            def read_json(path):
                with path.open("r", encoding="utf-8") as file:
                    return json.load(file)


            def find_split_image_dir(split, rows):
                sample_names = [row["model_input"]["image_name"] for row in rows[:10]]
                scores = Counter()
                for image_name in sample_names:
                    for candidate in EXTRACT_DIR.rglob(image_name):
                        if candidate.is_file():
                            scores[candidate.parent] += 1
                if not scores:
                    raise FileNotFoundError(f"{split} 이미지 디렉터리를 찾지 못했습니다.")
                best_dir, score = scores.most_common(1)[0]
                if score < min(3, len(sample_names)):
                    raise RuntimeError(f"{split} 이미지 경로 탐색이 불안정합니다: {scores}")
                return best_dir


            json_paths = {
                split: find_split_json(split)
                for split in ("train", "validation", "test")
            }
            records = {split: read_json(path) for split, path in json_paths.items()}
            image_dirs = {
                split: find_split_image_dir(split, records[split])
                for split in records
            }

            # AIVQA Dataset이 dataset_root/split/image_name으로 읽도록 링크합니다.
            for split, image_dir in image_dirs.items():
                target = DATASET_ROOT / split
                if target.is_symlink() and target.resolve() != image_dir.resolve():
                    target.unlink()
                elif target.exists() and not target.is_symlink():
                    raise RuntimeError(f"링크 대상 경로가 이미 디렉터리입니다: {target}")
                if not target.exists():
                    target.symlink_to(image_dir, target_is_directory=True)

            expected_counts = {"train": 1000, "validation": 200, "test": 800}
            rows = []
            for split, split_rows in records.items():
                missing = [
                    row["model_input"]["image_name"] for row in split_rows
                    if not (image_dirs[split] / row["model_input"]["image_name"]).is_file()
                ]
                forms = Counter(row["metadata"]["question_form"] for row in split_rows)
                assert len(split_rows) == expected_counts[split], (split, len(split_rows))
                assert not missing, (split, missing[:5])
                rows.append({
                    "split": split,
                    "rows": len(split_rows),
                    "MC": forms["MC"], "SA": forms["SA"], "LA": forms["LA"],
                    "missing_images": len(missing),
                    "image_dir": str(image_dirs[split]),
                })
            display(pd.DataFrame(rows))
            '''
        ),
    ]
)

nb.cells.extend(
    [
        md("### 5. 선택적 RAG·Kanana 안전 로더·SA 후처리 적용"),
        code(runtime_source),
        code(
            r'''
            print("적용된 핵심 함수 확인")
            print("- retriever:", rag_infer.QdrantRetriever.retrieve.__name__)
            print("- prompt:", rag_prompts.build_answer_feature.__name__)
            print("- retrieval cache:", rag_pipeline.retrieve_dataset_candidates.__name__)
            print("- validation:", type_train.evaluate_generation.__name__)
            print("- test generation:", rag_pipeline.generate_rag_predictions.__name__)
            print("- run dir:", RUN_DIR)
            '''
        ),
        md(
            """
            ### 6. 전체 학습 및 추론

            가장 오래 걸리는 셀입니다. 순서는 다음과 같습니다.

            1. Base Kanana로 train+validation 검색어 생성 및 selective RAG cache 저장
            2. Shared Adapter 2 epoch
            3. MC 3 / SA 5 / LA 3 epoch, 매 epoch 유형별 검증 최고 Adapter 저장
            4. Test 검색 및 유형별 최고 Adapter 추론
            5. `answer.json` 생성

            중단 후 다시 실행하면 완료된 검색 cache는 재사용됩니다. 학습 Adapter는
            Drive의 고정 RUN_DIR에 저장됩니다.
            """
        ),
        code(
            r'''
            if not DO_TRAIN:
                raise ValueError(
                    "이 노트북은 champion 전체 학습용입니다. DO_TRAIN=True로 실행하세요."
                )

            pipeline_args = argparse.Namespace(
                model_id=MODEL_ID,
                train_json=json_paths["train"],
                validation_json=json_paths["validation"],
                test_json=json_paths["test"],
                dataset_root=DATASET_ROOT,
                output_dir=RUN_DIR.parent,
                qdrant_path=LOCAL_QDRANT_DIR,
                qdrant_url=None,
                collection=QDRANT_COLLECTION,
                model_cache=MODEL_CACHE_DIR,
                score_threshold=RETRIEVAL_BASE_THRESHOLD,
                retrieval_page_size=100,
                rag_device="cpu",
                rag_fp32=False,
                max_rag_chars=max(RAG_CONTEXT_LIMITS.values()),
                search_max_new_tokens=64,
                train_batch_size=1,
                eval_batch_size=1,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                shared_learning_rate=SHARED_LEARNING_RATE,
                type_learning_rate=TYPE_LEARNING_RATE,
                shared_weight_decay=0.01,
                type_weight_decay=0.03,
                warmup_ratio=0.10,
                max_grad_norm=1.0,
                max_length=MAX_LENGTH,
                max_new_tokens=256,
                num_workers=0,
                seed=SEED,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                dtype=dtype_name,
                attn_implementation="sdpa",
                load_in_4bit=False,
                gradient_checkpointing=True,
            )

            answer_path = rag_pipeline.run_pipeline(pipeline_args)
            print("전체 파이프라인 완료:", answer_path)
            '''
        ),
        md("### 7. 유형별 최고 Validation 결과와 RAG 적용률 확인"),
        code(
            r'''
            selection_keys = {
                "MC": "mc_accuracy", "SA": "sa_exact_match", "LA": "descriptive_avg"
            }
            best_logs = {}
            for form, key in selection_keys.items():
                candidates = [
                    row for row in VALIDATION_EVAL_LOG
                    if row["question_form"] == form
                ]
                if not candidates:
                    raise RuntimeError(f"{form} validation 기록이 없습니다.")
                best_logs[form] = max(candidates, key=lambda row: row["metrics"][key])

            validation_rows = []
            for form, row in best_logs.items():
                for question_id, prediction, reference in zip(
                    row["question_ids"], row["predictions"], row["references"]
                ):
                    validation_rows.append({
                        "question_id": question_id,
                        "question_form": form,
                        "best_epoch_call": row["epoch_call"],
                        "prediction": prediction,
                        "reference": reference,
                    })
            validation_df = pd.DataFrame(validation_rows)
            validation_path = RUN_DIR / "validation_best_predictions.csv"
            validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")

            mc = best_logs["MC"]["metrics"]["mc_accuracy"]
            sa = best_logs["SA"]["metrics"]["sa_exact_match"]
            descriptive = best_logs["LA"]["metrics"]["descriptive_avg"]
            diagnostic_score = (mc + sa + descriptive) / 3
            diagnostic = {
                "warning": (
                    "Shared 단계가 train+validation을 사용하므로 완전한 홀드아웃 점수가 "
                    "아니며, 리더보드 점수를 보장하지 않습니다."
                ),
                "best_epoch_call": {
                    form: row["epoch_call"] for form, row in best_logs.items()
                },
                "mc_accuracy": mc,
                "sa_exact_match": sa,
                "descriptive_avg": descriptive,
                "diagnostic_score": diagnostic_score,
            }
            diagnostic_path = RUN_DIR / "validation_diagnostic.json"
            diagnostic_path.write_text(
                json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            display(pd.DataFrame([
                {
                    "metric": "accuracy", "value": mc * 100,
                    "best_epoch": best_logs["MC"]["epoch_call"],
                },
                {
                    "metric": "exact_match", "value": sa * 100,
                    "best_epoch": best_logs["SA"]["epoch_call"],
                },
                {
                    "metric": "descriptive_avg", "value": descriptive * 100,
                    "best_epoch": best_logs["LA"]["epoch_call"],
                },
                {"metric": "diagnostic_score", "value": diagnostic_score * 100},
            ]))
            print("검증 예측:", validation_path)
            print("주의:", diagnostic["warning"])
            '''
        ),
        code(
            r'''
            cache_rows = []
            cache_dir = RUN_DIR / "rag_cache"
            for cache_path in sorted(cache_dir.glob("*.json")):
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    rows = cached.get("rows", []) if isinstance(cached, dict) else cached
                    applied = sum(bool(row.get("candidates")) for row in rows)
                    reasons = Counter(
                        candidate.get("payload", {}).get("_retrieval", {}).get("reason", "")
                        for row in rows
                        for candidate in row.get("candidates", [])
                    )
                    cache_rows.append({
                        "cache": cache_path.name,
                        "samples": len(rows),
                        "rag_applied": applied,
                        "apply_rate": applied / len(rows) if rows else 0.0,
                        "reasons": dict(reasons),
                    })
                except Exception as error:
                    cache_rows.append({"cache": cache_path.name, "error": str(error)})
            rag_usage_df = pd.DataFrame(cache_rows)
            display(rag_usage_df)
            rag_usage_path = RUN_DIR / "rag_usage_summary.csv"
            rag_usage_df.to_csv(rag_usage_path, index=False, encoding="utf-8-sig")
            '''
        ),
        md("### 8. 제출 JSON·ZIP 및 실험 요약 생성"),
        code(
            r'''
            submission = json.loads(Path(answer_path).read_text(encoding="utf-8"))
            test_records = records["test"]
            assert isinstance(submission, list) and len(submission) == 800
            assert len(test_records) == len(submission)
            for source, predicted in zip(test_records, submission):
                assert str(source["metadata"]["question_id"]) == str(
                    predicted["metadata"]["question_id"]
                )
                answer = predicted.get("model_output", {}).get("answer")
                assert isinstance(answer, str) and answer.strip()

            submission_path = RUN_DIR / "submission_kanana15v_3b_unified_rag_champion.json"
            submission_path.write_text(
                json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            submission_zip = RUN_DIR / "submission_kanana15v_3b_unified_rag_champion.zip"
            with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(submission_path, arcname=submission_path.name)

            experiment_summary = {
                "run_name": RUN_NAME,
                "repo_url": REPO_URL,
                "repo_commit": REPO_COMMIT,
                "model_id": MODEL_ID,
                "model_revision": KANANA_REVISION,
                "rag_documents": len(normalized_rows),
                "rag_sha256": RAG_SHA256,
                "rag_sources": dict(source_counts),
                "rag_image_coverage": coverage,
                "rag_build_mode": RAG_BUILD_MODE,
                "retrieval_signature": RETRIEVAL_SIGNATURE,
                "shared_epochs": SHARED_EPOCHS,
                "type_epochs": TYPE_EPOCHS_BY_FORM,
                "learning_rates": {
                    "shared": SHARED_LEARNING_RATE,
                    "typed": TYPE_LEARNING_RATE,
                },
                "sa_length_mode": "free_generation_then_retry_only_hard_constraint",
                "validation_diagnostic": diagnostic,
                "submission_json": str(submission_path),
                "submission_zip": str(submission_zip),
            }
            summary_path = RUN_DIR / "champion_experiment_summary.json"
            summary_path.write_text(
                json.dumps(experiment_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("제출할 파일:", submission_path)
            print("편의용 ZIP:", submission_zip)
            print("실험 요약:", summary_path)
            '''
        ),
        md(
            """
            ### 결과 확인 순서

            1. `rag_usage_summary.csv`에서 RAG가 전체 문항에 강제되지 않았는지 확인
            2. `validation_diagnostic.json`에서 MC / SA / LA 최고 epoch 확인
            3. `validation_best_predictions.csv`에서 특히 SA 오답과 길이 준수 확인
            4. 리더보드에는 **JSON 파일만** 제출
               - `submission_kanana15v_3b_unified_rag_champion.json`
            5. 기존 최고점과 지표별 비교
               - 전체 44.8395
               - accuracy 73.7349
               - exact_match 25.7282
               - descriptive_avg 35.0555

            첫 결과가 47점에 못 미치더라도, 이 실행의 retrieval cache와 유형별
            validation 결과를 이용해 임계값·RAG 적용 대상만 바꾼 다음 실험을 설계할 수
            있습니다. 여러 설정을 동시에 바꾸지 말고 리더보드 비교 실험을 하나씩
            누적하세요.
            """
        ),
    ]
)

nb.cells.extend(
    [
        md("### 3. 통합 RAG JSONL 검증 및 선택적 이미지 복사"),
        code(
            r'''
            if not RAG_JSONL_PATH.is_file():
                candidates = list(DRIVE_PROJECT_DIR.rglob("unified_rag.jsonl"))
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        "Drive의 CV_korea/unified_rag_db/unified_rag.jsonl와 "
                        "그 옆 images/ 폴더를 준비하세요. 발견 후보: " + str(candidates)
                    )
                RAG_JSONL_PATH = candidates[0]
                RAG_BUNDLE_DIR = RAG_JSONL_PATH.parent


            def file_sha256(path):
                digest = hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            rag_rows = []
            invalid_lines = []
            with RAG_JSONL_PATH.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        invalid_lines.append((line_number, str(error)))
                        continue
                    if isinstance(value, dict):
                        rag_rows.append(value)

            required = {
                "doc_id", "source", "title", "search_terms", "description", "image_path"
            }
            assert not invalid_lines, invalid_lines[:5]
            assert len(rag_rows) > 0
            assert all(required.issubset(row) for row in rag_rows)
            assert len({str(row["doc_id"]) for row in rag_rows}) == len(rag_rows)

            source_counts = Counter(str(row["source"]) for row in rag_rows)
            declared = 0
            copied = 0
            missing = []
            normalized_rows = []
            for row in tqdm(rag_rows, desc="RAG 이미지 로컬 복사"):
                row = dict(row)
                raw_paths = row.get("image_path") or []
                if isinstance(raw_paths, str):
                    raw_paths = [raw_paths]
                normalized_paths = []
                for image_index, raw in enumerate(raw_paths):
                    declared += 1
                    raw_path = Path(str(raw)).expanduser()
                    source_path = (
                        raw_path if raw_path.is_absolute()
                        else RAG_JSONL_PATH.parent / raw_path
                    )
                    if not source_path.is_file():
                        missing.append(str(raw))
                        continue
                    if raw_path.is_absolute():
                        suffix = source_path.suffix.lower() or ".jpg"
                        relative = Path("images") / "imported" / (
                            f"{row['doc_id']}_{image_index}{suffix}"
                        )
                    else:
                        relative = raw_path
                    destination = LOCAL_RAG_DIR / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if (
                        not destination.is_file()
                        or destination.stat().st_size != source_path.stat().st_size
                    ):
                        shutil.copy2(source_path, destination)
                    normalized_paths.append(relative.as_posix())
                    copied += 1
                row["image_path"] = normalized_paths
                normalized_rows.append(row)

            coverage = copied / declared if declared else 1.0
            RAG_BUILD_MODE = "text_image" if copied else "text_only"
            if missing:
                print(
                    f"실제 파일이 없는 image_path {len(missing)}개는 README 규칙대로 "
                    "image vector만 생략합니다."
                )

            with LOCAL_RAG_JSONL.open("w", encoding="utf-8") as file:
                for row in normalized_rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

            RAG_SHA256 = file_sha256(LOCAL_RAG_JSONL)
            RETRIEVAL_SIGNATURE = {
                "rag_sha256": RAG_SHA256,
                "repo_commit": REPO_COMMIT,
                "thresholds": {
                    "base": RETRIEVAL_BASE_THRESHOLD,
                    "dual_text": DUAL_TEXT_THRESHOLD,
                    "dual_image": DUAL_IMAGE_THRESHOLD,
                    "strong_text": STRONG_TEXT_THRESHOLD,
                    "strong_image": STRONG_IMAGE_THRESHOLD,
                },
                "top_k": RAG_TOP_K,
                "context_limits": RAG_CONTEXT_LIMITS,
            }
            display(pd.DataFrame(source_counts.items(), columns=["source", "documents"]))
            print("RAG 문서:", len(normalized_rows))
            print("설명 없음:", sum(not str(row.get("description", "")).strip() for row in normalized_rows))
            print(f"이미지 복사: {copied}/{declared} ({coverage:.1%})")
            print("Qdrant 구축 모드:", RAG_BUILD_MODE)
            print("RAG SHA-256:", RAG_SHA256)
            '''
        ),
        md("### 4. KURE 중심 Qdrant DB 생성 또는 Drive cache 복원"),
        code(
            r'''
            qdrant_cache_dir = DRIVE_PROJECT_DIR / "qdrant_cache"
            qdrant_cache_dir.mkdir(parents=True, exist_ok=True)
            QDRANT_ARCHIVE = qdrant_cache_dir / (
                f"unified_{RAG_SHA256[:12]}_{REPO_COMMIT[:8]}_qdrant.zip"
            )
            QDRANT_COLLECTION = "aivqa_unified_rag"

            if LOCAL_QDRANT_DIR.exists():
                shutil.rmtree(LOCAL_QDRANT_DIR)

            if QDRANT_ARCHIVE.is_file() and not REBUILD_QDRANT:
                print("Drive의 동일 DB Qdrant cache를 복원합니다:", QDRANT_ARCHIVE)
                LOCAL_QDRANT_DIR.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(QDRANT_ARCHIVE) as archive:
                    archive.extractall(LOCAL_QDRANT_DIR)
            else:
                print(
                    "2,500건의 KURE text vector를 생성합니다. "
                    "실제 이미지가 있는 문서만 CLIP vector도 함께 저장합니다."
                )
                command = [
                    sys.executable,
                    str(REPO_DIR / "rag_db" / "build_qdrant.py"),
                    "--input", str(LOCAL_RAG_JSONL),
                    "--qdrant-path", str(LOCAL_QDRANT_DIR),
                    "--collection", QDRANT_COLLECTION,
                    "--model-cache", str(MODEL_CACHE_DIR),
                    "--device", "cuda",
                    "--batch-size", "64",
                    "--text-batch-size", "64",
                    "--image-batch-size", "32",
                    "--recreate",
                ]
                subprocess.check_call(command, cwd=str(REPO_DIR))

                local_archive_base = COLAB_ROOT / f"unified_{RAG_SHA256[:12]}_qdrant"
                local_zip = Path(
                    shutil.make_archive(
                        str(local_archive_base), "zip", root_dir=LOCAL_QDRANT_DIR
                    )
                )
                shutil.copy2(local_zip, QDRANT_ARCHIVE)
                print("다음 실행용 cache 저장:", QDRANT_ARCHIVE)

            assert LOCAL_QDRANT_DIR.is_dir()
            print("Qdrant 로컬 경로:", LOCAL_QDRANT_DIR)
            '''
        ),
    ]
)

# Section 3/4 was appended after the pipeline section to keep the builder source
# readable. Move that four-cell block before the runtime patch in the notebook.
assert len(nb.cells) == 26, len(nb.cells)
nb.cells = nb.cells[:11] + nb.cells[22:26] + nb.cells[11:22]

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(OUTPUT_PATH)
