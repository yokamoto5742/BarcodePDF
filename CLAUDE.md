# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

PDF内のCODE128バーコードを読み取り、その内容でファイル名を変更するWindows向けデスクトップアプリ。
tkinter GUI + 取込フォルダの定期走査（ポーリング）で常駐動作する。

## 環境とコマンド

Python 3.13 / 依存は uv 管理（`pyproject.toml` + `uv.lock`）。`pip install` や `requirements.txt` は使わない。

```bash
uv sync                                # 依存の同期
.venv\Scripts\python.exe main.py       # アプリ起動
.venv\Scripts\python.exe build.py      # PyInstallerでexe生成
```

テストコマンドは `.claude/rules/testing.md` を参照。

## 注意点
- pyzbar はネイティブのzbar DLLに依存する。バーコード読み取りの動作確認はWindows実機で行う。
