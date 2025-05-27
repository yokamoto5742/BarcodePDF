# BarcodePDF

## 概要

BarcodePDFは、PDFファイルからバーコードを自動的に読み取り、そのバーコード内容でファイル名を変更するPythonアプリケーションです。指定したフォルダを監視し、新しいPDFファイルが追加されると自動的に処理を実行します。

## 主な機能

- **自動バーコード検出**: PDFファイル内の画像からCODE128バーコードを検出
- **ファイル名自動変更**: 検出したバーコードの内容でPDFファイル名を変更
- **フォルダ監視**: 指定フォルダを監視し、新しいファイルを自動処理
- **GUI設定画面**: 直感的な設定インターフェース
- **エラー処理**: バーコードが検出できないファイルを別フォルダに移動
- **ログ機能**: 処理履歴とエラーログの自動記録
- **設定の永続化**: 設定内容をconfig.iniファイルに保存

## 必要な環境

### Python
- Python 3.11以上

### 必要なライブラリ
以下のライブラリをインストールする必要があります：

```bash
pip install opencv-python
pip install PyMuPDF
pip install Pillow
pip install pyzbar
pip install watchdog
```

### システム要件
- **Windows**: Windows 10 64ビット以上推奨
- 
## インストールと初期設定

1. **ファイルの準備**
   - `main.py`、`config.ini`をダウンロード
   - 同じフォルダに配置

2. **必要なフォルダの作成**
   ```
   C:\Shinseikai\BarcodePDF\
   ├── processing\     (処理対象フォルダ)
   ├── error\         (エラーファイル用)
   ├── log\           (ログファイル用)
   └── preprocessing\ (前処理用・オプション)
   ```

3. **設定ファイルの調整**
   - `config.ini`でフォルダパスを環境に合わせて修正

## 使い方

### 1. アプリケーションの起動

```bash
python main.py
```

### 2. 設定画面での操作

アプリケーション起動後、以下の設定が可能です：

- **処理フォルダ**: バーコード読み取り対象のPDFファイルを配置するフォルダ
- **エラーフォルダ**: バーコードが検出できないファイルの移動先
- **完了フォルダ**: 処理済みファイルの移動先
- **ログフォルダ**: ログファイルの保存先
- **エラーフォルダを自動的に開く**: エラー発生時にフォルダを自動で開く

### 3. PDF処理の流れ

1. PDFファイルを**処理フォルダ**に配置
2. アプリが自動的にファイルを検出
3. バーコード読み取り処理を実行
4. 成功時：バーコード内容でファイル名を変更し**完了フォルダ**に移動
5. 失敗時：元のファイル名で**エラーフォルダ**に移動

### 4. ステータス確認

アプリ画面下部のステータス表示で、以下の情報を確認できます：
- 処理中のファイル情報
- 成功/失敗の結果
- エラーメッセージ

## 設定ファイル（config.ini）の詳細

```ini
[Directories]
processing_dir = C:\Shinseikai\BarcodePDF\processing
error_dir = C:\Shinseikai\BarcodePDF\error
done_dir = C:\pdfkarte\TmpPdf
log_dir = C:\Shinseikai\BarcodePDF\log

[UI]
width = 600
height = 500

[Options]
auto_open_error_folder = True
start_minimized = True

[Logging]
retention_days = 14
```

### 設定項目の説明

- `processing_dir`: 処理対象PDFファイルの監視フォルダ
- `error_dir`: エラーファイルの保存先
- `done_dir`: 処理済みファイルの保存先
- `log_dir`: ログファイルの保存先
- `width`/`height`: アプリウィンドウのサイズ
- `auto_open_error_folder`: エラー時のフォルダ自動表示
- `start_minimized`: 最小化で起動
- `retention_days`: ログファイルの保持日数

## ログ機能

- **ログファイル**: `BarcodePDF.log`として日次ローテーション
- **保持期間**: 設定で指定した日数（デフォルト14日）
- **記録内容**:
  - ファイル処理の開始/完了
  - バーコード検出結果
  - エラー情報
  - 設定変更履歴

## トラブルシューティング

### よくある問題

**Q: バーコードが検出されない**
- PDFの解像度が低い場合があります
- バーコードがCODE128形式か確認してください
- 画像の品質を向上させてから再試行

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
main.py
├── Config クラス         : 設定管理
├── PDFProcessorApp クラス : メインアプリケーション
├── PDFHandler クラス     : ファイルシステム監視
└── 各種関数
    ├── extract_images_from_pdf : PDF画像抽出
    ├── read_barcode_from_pdf   : バーコード読み取り
    ├── process_pdf             : PDF処理メイン
    └── setup_logger           : ログ設定
```

### 主要なライブラリ

- **fitz (PyMuPDF)**: PDF処理
- **pyzbar**: バーコード読み取り
- **opencv-python**: 画像処理
- **watchdog**: ファイル監視
- **tkinter**: GUI

### カスタマイズ

バーコード形式を変更する場合：
```python
# read_barcode_from_pdf 関数内
barcodes = decode(gray, symbols=[ZBarSymbol.CODE128])
# 他の形式: ZBarSymbol.CODE39, ZBarSymbol.QRCODE など
```

画像処理の調整：
```python
# コントラスト強調の値を変更
enhancer.enhance(2.0)  # 2.0を他の値に変更
```

## ライセンス

このプロジェクトはオープンソースです。詳細については、プロジェクトのライセンスファイルを参照してください。

## サポート

技術的な問題や機能要求については、プロジェクトのIssuesページで報告してください。
