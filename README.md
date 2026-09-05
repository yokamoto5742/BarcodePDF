# BarcodePDF

## 概要

BarcodePDFは、PDFファイルからバーコードを自動的に読み取り、そのバーコード内容でファイル名を変更するPythonアプリケーションです。指定したフォルダを監視し、新しいPDFファイルが追加されると自動的に処理を実行します。

## 主な機能

- **自動バーコード検出**: PDFファイルの上部帯からCODE128バーコードを自動検出（高解像度処理）
- **ファイル名自動変更**: 検出したバーコードの内容でPDFファイル名を変更
- **フォルダ監視**: 指定フォルダを監視し、新しいファイルを自動処理
- **GUI設定画面**: 直感的な設定インターフェース
- **エラー処理**: バーコードが検出できないファイルを別フォルダに移動
- **エラーPDF自動削除**: エラーフォルダに溜まったPDFを保持期間に応じて自動削除
- **ログ機能**: 処理履歴とエラーログの自動記録
- **設定の永続化**: 設定内容をconfig.iniファイルに保存

## 必要な環境

### Python
- Python 3.13 以上

### 必要なライブラリ
依存関係は `pyproject.toml` と `uv.lock` で管理しています。[uv](https://docs.astral.sh/uv/) で一括インストールしてください：

```bash
uv sync
```

主な依存: opencv-python-headless / PyMuPDF / Pillow / pyzbar / numpy

### システム要件
- **Windows**: Windows 10 64ビット以上推奨

## インストールと初期設定

1. **ファイルの準備**
   - リポジトリをクローンし、`uv sync` で依存をインストール
   - `utils/config.ini` が同梱されていることを確認

2. **必要なフォルダの作成**
   ```
   C:\Shinseikai\BarcodePDF\
   ├── preprocessing\ (取込フォルダ・スキャナーの出力先)
   ├── error\         (エラーファイル用)
   └── log\           (ログファイル用)
   ```

3. **設定ファイルの調整**
   - `utils/config.ini`でフォルダパスを環境に合わせて修正

## 使い方

### 1. アプリケーションの起動

```bash
uv run python main.py
```

### 2. 設定画面での操作

アプリケーション起動後、以下の設定が可能です：

- **取込フォルダ**: スキャナーがPDFを出力するフォルダ（バーコード読み取りの対象）
- **エラーフォルダ**: バーコードが検出できないファイルの移動先
- **完了フォルダ**: 処理済みファイルの移動先
- **ログフォルダ**: ログファイルの保存先
- **エラーフォルダを自動的に開く**: エラー発生時にフォルダを自動で開く

### 3. PDF処理の流れ

**アプリ起動時**:
- エラーフォルダの保持期間を超過したPDFを自動削除（`log_retention_days`設定値を使用）
- GUIウィンドウを表示し、フォルダ監視を開始

**常駐動作中**:
1. スキャナーがPDFファイルを**取込フォルダ**に出力
2. アプリが2秒ごとに取込フォルダを走査し、書き込みが完了したPDFにバーコード読み取り処理を実行
3. 成功時：バーコード内容でファイル名を変更し**完了フォルダ**に移動
4. 失敗時：元のファイル名で**エラーフォルダ**に移動

取込フォルダにPDFを直接配置しても処理されます。

取込フォルダの検出はファイル更新イベントではなく、サイズと更新日時が前回の走査から
変化していないことの確認によって行います。スキャン直後にファイル名を変更するスキャナーでも、
書き込み途中のファイルを読んでしまうことなく処理できます。

### 4. ステータス確認

アプリ画面下部のステータス表示で、以下の情報を確認できます：
- 処理中のファイル情報
- 成功/失敗の結果
- エラーメッセージ

## 設定ファイル（config.ini）の詳細

```ini
[Directories]
target_dir = C:\Shinseikai\BarcodePDF\preprocessing
error_dir = C:\Shinseikai\BarcodePDF\error
done_dir = C:\pdfkarte\TmpPdf

[UI]
width = 600
height = 500

[Barcode]
contrast_factor = 2.0
render_zoom = 2.5
top_band_ratio = 0.15
min_barcode_width_ratio = 0.20

[Options]
auto_open_error_folder = True

[LOGGING]
log_directory = C:\Shinseikai\BarcodePDF\log
log_retention_days = 7
log_level = INFO
project_name = BarcodePDF
```

### 設定項目の説明

- `target_dir`: スキャナーの出力先フォルダ（バーコード読み取りの対象）
- `error_dir`: エラーファイルの保存先
- `done_dir`: 処理済みファイルの保存先
- `width`/`height`: アプリウィンドウのサイズ
- `contrast_factor`: 読み取り前にかけるコントラスト強調の倍率
- `render_zoom`: PDFページの描画拡大率）
- `top_band_ratio`: ページ上端からバーコードを探す高さの割合
- `min_barcode_width_ratio`: 採用するバーコードの、ページ幅に対する最小幅
- `auto_open_error_folder`: エラー時のフォルダ自動表示
- `log_directory`: ログファイルの保存先
- `log_retention_days`: ログファイルおよびエラーフォルダのPDFの保持日数（アプリ起動時に超過ファイルを削除）
- `log_level`: ログ出力レベル（DEBUG/INFO/WARNING/ERROR）

## ログ機能

- **ログファイル**: `BarcodePDF.log`として日次ローテーション
- **保持期間**: 設定で指定した日数（デフォルト14日）
- **記録内容**:
  - ファイル処理の開始/完了
  - バーコード検出結果
  - エラー情報
  - 設定変更履歴
  - トレース行（どのファイルをどこへ送ったか）

### トレースログ

処理を終えたファイルごとに、移動元・バーコード・移動先を1行で記録します。

```
2026-09-01 18:30:12,345 - service.pdf_processor - INFO - TRACE result=SUCCESS src=C:\Shinseikai\BarcodePDF\preprocessing\scan001.pdf barcode=1234567890 dst=C:\pdfkarte\TmpPdf\1234567890.pdf
2026-09-01 18:31:05,120 - service.pdf_processor - INFO - TRACE result=NO_BARCODE src=C:\Shinseikai\BarcodePDF\preprocessing\scan002.pdf barcode= dst=C:\Shinseikai\BarcodePDF\error\scan002.pdf
```

- `result`: `SUCCESS`（完了フォルダへ移動）/ `NO_BARCODE`（バーコード未検出）/ `ERROR`（処理中に例外）
- `src`: 移動元のフルパス
- `barcode`: 読み取れたバーコード内容（読めなかった場合は空）
- `dst`: 移動先のフルパス（移動できなかった場合は空）

`TRACE` で絞り込めば、処理したファイルの送り先を一覧できます。

```bash
findstr TRACE C:\Shinseikai\BarcodePDF\log\BarcodePDF.log
```

## トラブルシューティング

### よくある問題

**Q: バーコードが検出されない**
- バーコードは必ずCODE128形式である必要があります（QRコード等には非対応）
- バーコードはPDFページの上部15%以内に配置してください（下部のバーコードは読み込まれません）
- バーコード幅がページ幅の20%以上である必要があります（細いバーコードは検出されません）
- PDFが破損していないか確認してください
- 複数のバーコードがある場合は、最も幅の広いものが採用されます

**Q: フォルダが監視されない**
- フォルダパスが正しいか確認
- フォルダの読み書き権限を確認
- アプリケーションを管理者権限で実行

**Q: エラーフォルダが開かない**
- Windowsの場合：エクスプローラーが正常に動作するか確認
- macOS/Linuxの場合：openコマンドが利用可能か確認

### エラーコードと対処法

- **ファイルが見つかりません**: パスの指定を確認
- **バーコード読み取り中にエラー**: PDFファイルの破損を確認
- **ファイルの移動中にエラー**: ディスク容量と権限を確認

## 開発者向け情報

### アーキテクチャ

```
main.py                       : エントリポイント（起動時のクリーンアップ → GUI起動）
├── app/
│   ├── __init__.py           : __version__（GUIのバージョン表示元）
│   └── main_window.py        : PDFProcessorApp（GUIとフォルダ監視の起動）
├── service/
│   ├── barcode_reader.py     : PDFの上部帯をレンダリングしてCODE128読み取り
│   │                           (高解像度・ノイズ除去・最大幅優先)
│   ├── pdf_processor.py      : process_pdf（振り分けとトレースログ）
│   ├── pdf_watcher.py        : PdfWatcher（取込フォルダのポーリング監視）
│   └── error_pdf_cleanup.py  : cleanup_error_pdfs（エラーフォルダの古いPDF削除）
└── utils/
    ├── config_manager.py     : ConfigManager / AppConfig（config.ini の読み書き）
    ├── log_rotation.py       : setup_logging（日次ローテーションと古いログの削除）
    ├── file_cleanup.py       : delete_files_older_than（保持期間超過ファイルの削除）
    └── constants.py          : UI・ログメッセージの定数
```

### 主要なライブラリ

- **fitz (PyMuPDF)**: PDF処理
- **pyzbar**: バーコード読み取り
- **opencv-python**: 画像処理
- **tkinter**: GUI

### カスタマイズ

バーコード読み取り処理の詳細設定は `service/barcode_reader.py` 内の定数を調整してください：

```python
CONTRAST_FACTOR = 2.0           # コントラスト強調の度合い（高いほど細いバーが強調される）
RENDER_ZOOM = 3.0               # PDFレンダリング時の拡大率
TOP_BAND_RATIO = 0.15           # ページ上端から探索する高さの割合（15%）
MIN_BARCODE_WIDTH_RATIO = 0.20  # 採用するバーコードの最小幅（ページ幅の20%以上）
```

読み取り対象のバーコード形式は現在 **CODE128 のみ** です。他の形式に対応させる場合は、`_decode_code128` 関数を修正してください。

## ライセンス

このプロジェクトのライセンス情報については、 [LICENSE](docs/LICENSE) を参照してください。

## 更新履歴

更新履歴は [CHANGELOG.md](docs/CHANGELOG.md) を参照してください
