## 비타민 26-여름세션 CV 프로젝트
### 외국인을 위한 한국문화 멀티모달 RAG 기반 질의응답 시스템

## 환경 설정

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name ai-cv --display-name "Python (.venv AI_CV)"
```

## EDA 및 전처리

- 실행 결과 포함 노트북: `notebooks/01_eda_preprocessing.ipynb`
- Notion 정리본: `docs/notion-eda-summary.md`

노트북을 다시 생성하고 실행하려면 다음 명령을 사용합니다.

```bash
python scripts/build_eda_notebook.py
python scripts/execute_notebook.py notebooks/01_eda_preprocessing.ipynb
```

전처리 파생 데이터와 품질 검토 큐는 Git에서 제외된 `outputs/` 아래에 생성됩니다.
