# API (FastAPI)

## Run
```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

## Tests
```bash
pip install -e .[test]
pytest
```
