## テスト実行コマンド

```bash
# 全件
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short

# 単一ファイル
.venv\Scripts\python.exe -m pytest tests/service/test_barcode_reader.py -v

# 単一テスト
.venv\Scripts\python.exe -m pytest tests/service/test_barcode_reader.py::test_read_code128 -v

# カバレッジ付き
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short --cov=app --cov=service --cov=utils --cov-report=html
```
