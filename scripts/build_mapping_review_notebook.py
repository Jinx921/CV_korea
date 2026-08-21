from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "mapping_review_visual.ipynb"


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "colab": {"name": OUTPUT_PATH.name, "provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.x"},
}


def markdown(source: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    """
## 이미지·텍스트 RAG 매핑 후보 수동 검수

이 노트북은 `mapping_review.csv`의 후보를 눈으로 확인하고 `yes/no`를 기록합니다.

- 왼쪽: 대회 train·validation·test 원본 이미지
- 오른쪽: 매핑 후보 TourAPI 이미지
- 함께 표시: 질문, 선택지, 후보 제목·설명, 이미지·텍스트 유사도 근거
- 저장: 버튼을 누를 때마다 Google Drive의 CSV에 즉시 반영

판정 기준은 **두 이미지가 같은 장소·유물·음식·대상임을 확신할 수 있는지**입니다.
단순히 비슷한 종류의 이미지라면 `no`로 판정하세요. 이 작업에는 GPU가 필요하지 않습니다.
"""
)


markdown("### 1. Google Drive 연결과 경로 설정")


code(
    r'''
from pathlib import Path

from google.colab import drive


drive.mount("/content/drive")

DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/CV_korea")
DRIVE_ZIP_DIR = DRIVE_PROJECT_DIR / "data"
RAG_DB_DIR = DRIVE_PROJECT_DIR / "tourapi_image_db"
COMBINED_DIR = RAG_DB_DIR / "mapping" / "combined"
REVIEW_CSV = COMBINED_DIR / "mapping_review.csv"
QUERY_IMAGE_DIR = Path("/content/mapping_review_query_images")

COMBINED_DIR.mkdir(parents=True, exist_ok=True)
QUERY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("검수 CSV:", REVIEW_CSV)
print("대회 ZIP 폴더:", DRIVE_ZIP_DIR)
print("TourAPI DB:", RAG_DB_DIR)
print()
print("mapping_review.csv가 위 경로에 없다면 로컬 파일을 선택해 업로드합니다.")
'''
)


markdown("### 2. 검수 CSV 불러오기")


code(
    r'''
import shutil

import pandas as pd
from google.colab import files


if not REVIEW_CSV.is_file():
    uploaded = files.upload()
    if "mapping_review.csv" not in uploaded:
        raise FileNotFoundError("mapping_review.csv를 선택해야 합니다.")
    temporary = Path("/content/mapping_review.csv")
    temporary.write_bytes(uploaded["mapping_review.csv"])
    shutil.copy2(temporary, REVIEW_CSV)
    print("Drive에 저장했습니다:", REVIEW_CSV)

review_df = pd.read_csv(
    REVIEW_CSV,
    encoding="utf-8-sig",
    dtype={"question_id": str, "proposed_doc_id": str},
    keep_default_na=False,
)

required_columns = {
    "image_key", "split", "image_name", "question", "options",
    "proposed_doc_id", "proposed_title", "proposed_description",
    "proposed_image_path", "proposed_image_url", "review_decision",
    "review_note",
}
missing = required_columns - set(review_df.columns)
if missing:
    raise RuntimeError(f"검수 CSV에 필수 열이 없습니다: {sorted(missing)}")
if review_df["image_key"].duplicated().any():
    raise RuntimeError("검수 CSV의 image_key가 중복됩니다.")

review_df["review_decision"] = review_df["review_decision"].str.strip().str.lower()
invalid_decisions = set(review_df["review_decision"]) - {"", "yes", "no"}
if invalid_decisions:
    raise RuntimeError(f"review_decision에는 yes/no만 사용할 수 있습니다: {invalid_decisions}")

print("전체 후보:", len(review_df))
print("승인(yes):", int((review_df["review_decision"] == "yes").sum()))
print("거절(no):", int((review_df["review_decision"] == "no").sum()))
print("미검수:", int((review_df["review_decision"] == "").sum()))
'''
)


markdown("### 3. 대회 ZIP에서 검수 대상 원본 이미지만 추출")


code(
    r'''
import shutil
import zipfile


def resolve_archive(split: str) -> Path:
    exact = DRIVE_ZIP_DIR / f"{split}.zip"
    if exact.is_file():
        return exact
    matches = [
        path for path in DRIVE_ZIP_DIR.glob("*.zip")
        if path.name.lower() == f"{split}.zip"
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"{split}.zip을 하나로 찾지 못했습니다: {matches}")
    return matches[0]


query_image_paths = {}
for split, group in review_df.groupby("split", sort=False):
    archive_path = resolve_archive(str(split))
    with zipfile.ZipFile(archive_path) as archive:
        names_by_basename = {}
        for member_name in archive.namelist():
            if member_name.endswith("/"):
                continue
            names_by_basename.setdefault(Path(member_name).name, []).append(member_name)

        for image_name in group["image_name"].tolist():
            members = names_by_basename.get(str(image_name), [])
            if len(members) != 1:
                raise RuntimeError(
                    f"{split}:{image_name}을 ZIP 안에서 하나로 찾지 못했습니다: {members}"
                )
            target = QUERY_IMAGE_DIR / str(split) / str(image_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size == 0:
                with archive.open(members[0]) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            query_image_paths[f"{split}:{image_name}"] = target

if len(query_image_paths) != len(review_df):
    raise RuntimeError(
        f"원본 이미지 수가 후보 수와 다릅니다: {len(query_image_paths)} != {len(review_df)}"
    )

print("검수 대상 원본 이미지 준비 완료:", len(query_image_paths))
'''
)


markdown("### 4. 원본·후보 이미지와 텍스트를 함께 보며 판정")


code(
    r'''
import ast
import html
import io
import json
from urllib.parse import urlparse

import ipywidgets as widgets
import matplotlib.pyplot as plt
import requests
from IPython.display import HTML, clear_output, display
from PIL import Image, ImageOps


def save_review_csv() -> None:
    temporary = REVIEW_CSV.with_suffix(".csv.tmp")
    review_df.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(REVIEW_CSV)


def parse_options(value):
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [text]


def load_rgb(path: Path | None = None, url: str = "") -> Image.Image:
    if path is not None and path.is_file():
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    if url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: path={path}, url={url}")


def resolve_candidate_path(row) -> Path | None:
    raw = str(row["proposed_image_path"]).strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else RAG_DB_DIR / path


def compact_text(value, max_chars=2500):
    text = " ".join(str(value).split())
    return text if len(text) <= max_chars else text[:max_chars] + " …"


state = {"index": 0, "pending_only": True}
pending_indices = review_df.index[review_df["review_decision"] == ""].tolist()
if pending_indices:
    state["index"] = int(pending_indices[0])

output = widgets.Output()
note = widgets.Textarea(
    placeholder="선택 사항: 판단 근거 또는 애매한 점",
    description="메모",
    layout=widgets.Layout(width="100%", height="70px"),
)

previous_button = widgets.Button(description="← 이전", button_style="")
next_button = widgets.Button(description="다음 →", button_style="")
yes_button = widgets.Button(description="같은 대상 · YES", button_style="success")
no_button = widgets.Button(description="다른 대상 · NO", button_style="danger")
pending_button = widgets.Button(description="판정 지우기", button_style="warning")
download_button = widgets.Button(description="CSV 다운로드", button_style="info")


def candidate_indices():
    if state["pending_only"]:
        pending = review_df.index[review_df["review_decision"] == ""].tolist()
        return pending if pending else review_df.index.tolist()
    return review_df.index.tolist()


def move(step: int) -> None:
    indices = candidate_indices()
    current = state["index"]
    if current not in indices:
        state["index"] = int(indices[0])
    else:
        position = indices.index(current)
        state["index"] = int(indices[(position + step) % len(indices)])
    render()


def render() -> None:
    index = state["index"]
    row = review_df.loc[index]
    note.value = str(row["review_note"])

    with output:
        clear_output(wait=True)
        decided = review_df["review_decision"] != ""
        decision = str(row["review_decision"]) or "미검수"
        decision_color = {"yes": "#138a36", "no": "#c62828"}.get(decision, "#666")
        display(HTML(
            f"<h3>{index + 1}/{len(review_df)} · {html.escape(str(row['image_key']))}</h3>"
            f"<p><b>현재 판정:</b> <span style='color:{decision_color}'>{decision}</span> · "
            f"완료 {int(decided.sum())}/{len(review_df)}</p>"
        ))

        query_image = load_rgb(query_image_paths[str(row["image_key"])])
        candidate_path = resolve_candidate_path(row)
        candidate_image = load_rgb(
            candidate_path,
            str(row["proposed_image_url"]).strip(),
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].imshow(query_image)
        axes[0].set_title("대회 원본 이미지", fontsize=15)
        axes[1].imshow(candidate_image)
        axes[1].set_title(
            f"TourAPI 후보: {row['proposed_title']} ({row['proposed_doc_id']})",
            fontsize=15,
        )
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        plt.show()

        options = parse_options(row["options"])
        options_html = "".join(
            f"<li>{html.escape(str(option))}</li>" for option in options
        ) or "<li>선택지 없음</li>"
        matched_tokens = html.escape(str(row.get("matched_text_tokens", "")))
        display(HTML(f"""
        <div style="font-size:15px; line-height:1.65">
          <p><b>유형:</b> {html.escape(str(row.get('question_form', '')))}</p>
          <p><b>질문:</b> {html.escape(str(row['question']))}</p>
          <p><b>선택지:</b></p><ol>{options_html}</ol>
          <hr>
          <p><b>후보 제목:</b> {html.escape(str(row['proposed_title']))}</p>
          <p><b>후보 설명:</b> {html.escape(compact_text(row['proposed_description']))}</p>
          <p><b>근거 점수:</b>
             image={html.escape(str(row.get('image_score', '')))},
             margin={html.escape(str(row.get('image_margin', '')))},
             image_rank={html.escape(str(row.get('image_rank', '')))},
             text_rank={html.escape(str(row.get('text_rank', '')))},
             text_score={html.escape(str(row.get('text_score', '')))}</p>
          <p><b>일치 텍스트 토큰:</b> {matched_tokens}</p>
        </div>
        """))


def set_decision(decision: str) -> None:
    index = state["index"]
    review_df.at[index, "review_decision"] = decision
    review_df.at[index, "review_note"] = note.value.strip()
    save_review_csv()
    remaining = review_df.index[review_df["review_decision"] == ""].tolist()
    if remaining:
        state["index"] = int(remaining[0])
    render()


previous_button.on_click(lambda _: move(-1))
next_button.on_click(lambda _: move(1))
yes_button.on_click(lambda _: set_decision("yes"))
no_button.on_click(lambda _: set_decision("no"))
pending_button.on_click(lambda _: set_decision(""))


def download_csv(_):
    save_review_csv()
    files.download(str(REVIEW_CSV))


download_button.on_click(download_csv)

controls = widgets.HBox([
    previous_button, yes_button, no_button, pending_button, next_button, download_button
])
display(controls, note, output)
render()
'''
)


markdown("### 5. 검수 완료 여부 확인")


code(
    r'''
save_review_csv()

yes_count = int((review_df["review_decision"] == "yes").sum())
no_count = int((review_df["review_decision"] == "no").sum())
pending_count = int((review_df["review_decision"] == "").sum())

print("승인(yes):", yes_count)
print("거절(no):", no_count)
print("미검수:", pending_count)
print("저장 위치:", REVIEW_CSV)

if pending_count:
    pending_keys = review_df.loc[
        review_df["review_decision"] == "", "image_key"
    ].tolist()
    print("아직 검수하지 않은 항목:", pending_keys)
else:
    print("검수 완료. CSV를 로컬 프로젝트의 outputs/mapping/combined에 덮어쓴 뒤")
    print("python3 scripts/finalize_rag_mapping.py 를 실행하세요.")
'''
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT_PATH)
print(OUTPUT_PATH)
