# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

PDF内のCODE128バーコードを読み取り、その内容でファイル名を変更するWindows向けデスクトップアプリ。
tkinter GUI + watchdog によるフォルダ監視で常駐動作する。

## 環境とコマンド

Python 3.13 / 依存は uv 管理（`pyproject.toml` + `uv.lock`）。`pip install` や `requirements.txt` は使わない。

```bash
uv sync                                # 依存の同期
.venv\Scripts\python.exe main.py       # アプリ起動
.venv\Scripts\python.exe build.py      # PyInstallerでexe生成
```

テストコマンドは `.claude/rules/testing.md` を参照。

## 構成方針

現状ロジックは `main.py` に集約されているが、以下へ分割していく途中：

- `app/` — GUI・アプリケーション層（`PDFProcessorApp` など）
- `service/` — PDF処理・バーコード読み取りのドメインロジック
- `utils/` — 設定管理（`config_manager.py`）、ログ（`log_rotation.py`）

新規コードは `main.py` に足さず、上記のいずれかに置く。

## 注意点

- **config.ini の読み込み経路が2系統ある**。`utils/config_manager.py` は `sys._MEIPASS` を見てPyInstaller凍結時にも解決する正しい実装。`main.py` の `Config` クラスはカレントディレクトリの `'config.ini'` を直読みしており、起動ディレクトリ依存で壊れる。設定周りを触るときは `config_manager.py` 側に寄せる。
- **`build.py` は未定義変数 `new_version` を参照しており、そのままでは NameError で落ちる**（成果物自体は生成される）。
- **バージョン番号が2箇所にある**。`main.py` の `VERSION` / `LAST_UPDATED` と `pyproject.toml` の `version` が乖離しているため、どちらを更新するか確認する。
- `utils/log_rotation.py` の `project_name` デフォルトが `'VoiceScribe'` のまま（他プロジェクトからの流用）。
- pyzbar はネイティブのzbar DLLに依存する。バーコード読み取りの動作確認はWindows実機で行う。
