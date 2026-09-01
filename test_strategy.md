# BarcodePDF ユニットテスト戦略

対象コミット: `c1853a7` / 対象範囲: `app/`, `service/`, `utils/`, `main.py`
テスト基盤: pytest 8.4 + pytest-mock + pytest-cov（`pyproject.toml` の dev グループに導入済み、`testpaths = ["tests"]`）

---

## 1. コード構造の分析結果

### 1.1 モジュール構成と依存関係

```
main.py
 ├─ utils.log_rotation.setup_logging
 └─ app.main_window.PDFProcessorApp        ← tkinter GUI / watchdog Observer
      ├─ utils.config_manager.AppConfig     ← config.ini (configparser)
      ├─ service.pdf_processor
      │    ├─ PDFHandler (watchdog イベント)
      │    └─ process_pdf ─ service.barcode_reader.read_barcode_from_pdf
      │                       ├─ pymupdf (PDF→画像)
      │                       ├─ PIL / OpenCV (前処理)
      │                       └─ pyzbar (CODE128 デコード) ※ネイティブDLL依存
      └─ utils.log_rotation.setup_logging
```

依存の向きは単方向（`app` → `service` → `utils`）で循環はない。ただし以下の結合がテスト設計上の制約になる。

| 箇所 | 結合の種類 | テスト上の扱い |
|---|---|---|
| `barcode_reader` → pymupdf / pyzbar / cv2 | 外部ネイティブライブラリ | モジュール属性を `mocker.patch` で差し替え。実DLLを使う経路は統合テスト（Windows実機）に隔離 |
| `pdf_processor` → `shutil` / `os` / `subprocess` / `os.startfile` | ファイルシステム・OS | `tmp_path` による実ファイル操作を基本とし、OS固有API（`os.startfile`）のみモック |
| `pdf_processor` → `AppConfig` | 具象クラスへの直接依存（DIなし） | 属性のみ参照するため軽量スタブ（`SimpleNamespace` 相当のフェイク）で代替可能 |
| `main_window` → tkinter / watchdog Observer | GUIランタイム・スレッド | `PDFProcessorApp` 全体をインスタンス化せず、メソッド単体を `__new__` + 属性注入で検証。あるいは `tk`/`ttk`/`Observer` を丸ごとモック |
| `log_rotation.setup_logging` → ルートロガー | グローバル状態の破壊的変更 | **必須**: ルートロガーのハンドラを保存・復元する autouse fixture を用意（未対応だと pytest 自身のログ捕捉が壊れる） |

### 1.2 複雑度の評価

| モジュール | 行数 | 分岐の多さ | 副作用 | 総合 |
|---|---|---|---|---|
| `service/pdf_processor.py` | 153 | 高（成功／バーコード無し／例外／二重例外の4経路） | ファイル移動・削除相当 | **最重要** |
| `service/barcode_reader.py` | 86 | 中（画像形状3分岐 + デコード2段リトライ） | なし（読み取りのみ） | 高 |
| `utils/config_manager.py` | 104 | 中（エンコーディングfallback、型別fallback） | 設定ファイル書き込み | 高 |
| `utils/log_rotation.py` | 171 | 中（正規表現マッチ＋保持期間判定） | ログファイル削除 | 中 |
| `app/main_window.py` | 180 | 低〜中（GUIイベント配線） | GUI・スレッド | 中 |
| `utils/constants.py` | 45 | なし | なし | 対象外 |

### 1.3 エラーハンドリングの実装状況と設計上のリスク

コード読解で確認した、テストで固定すべき（あるいはテストによって露見させるべき）挙動。R1〜R3・R6〜R9 は本戦略の実装にあわせて修正済み。

| # | 箇所 | 内容 | 影響 | 対応 |
|---|---|---|---|---|
| R1 | `pdf_processor._move_to_done_dir` | バーコード文字列を**無検証で**ファイル名に使用。CODE128 は `\`, `/`, `:`, `..` を符号化可能 | 完了フォルダ外への書き出し（パストラバーサル）／`OSError` | **修正済**: `is_valid_barcode` を追加し、不正なバーコードはエラーフォルダへ（TRACE=INVALID_BARCODE） |
| R2 | 同上 `shutil.move` | 同名ファイルが `done_dir` に既存の場合、Windowsでは copy2+unlink 経路で**無警告上書き** | 先着PDFの消失 | **修正済**: 上書きを仕様として確定。移動前に既存ファイルを削除し、意図を明示（エラーフォルダも同様） |
| R3 | `barcode_reader.extract_images_from_pdf` | `pdf_document.close()` が `finally` にない。破損PDFで例外時にハンドルがリーク | 監視ディレクトリのファイルがロックされ続ける | **修正済**: `with pymupdf.open(...)` に変更 |
| R4 | `barcode_reader.read_barcode_from_pdf` | `extract_images_from_pdf` は try の外。PDF展開失敗は `process_pdf` の包括 except まで伝播し、当該PDFはエラーフォルダへ | 仕様として妥当 | 仕様のままテストで明文化 |
| R5 | `pdf_processor.process_pdf` | 包括 `except Exception` → `_handle_failed_file`。`_move_to_done_dir` が move 成功後に失敗した場合、ファイルは既に存在せず TRACE=ERROR のみ記録される | 追跡ログの意味が経路で変わる | 仕様のままテストで明文化 |
| R6 | `config_manager.save_config` | 例外メッセージが `"Failed to load config"`（save の誤り） | 障害調査時の誤誘導 | **修正済**: `"Failed to save config"` |
| R7 | `config_manager.AppConfig.save` | セクション欠損時 `KeyError`。`_ensure_section` が定義済みだが**未使用（デッドコード）** | 保存失敗 | **修正済**: `ensure_section` に改名し `AppConfig.save` から使用 |
| R8 | `log_rotation.setup_logging` | `getattr(logging, log_level.upper())` は `AttributeError` のみ捕捉。`log_level=Formatter` のような実在する非int属性では `setLevel` が `TypeError` → 包括 except で「初期化中にエラー」に化ける | 起動不能 | **修正済**: `getattr(..., None)` + `isinstance(level, int)` 判定 |
| R9 | `barcode_reader._to_grayscale_array` | 呼び出し元 `_to_enhanced_grayscale` が必ず `'L'`（2次元）へ変換するため、3次元分岐は**到達不能** | 到達不能コード | **修正済**: 関数を削除し `np.array(image)` を直接使用 |

**確定した仕様（R1・R2）**

- **不正バーコードはエラーフォルダ行き**。`is_valid_barcode` が偽を返す条件: 空文字／200文字超／Windowsの禁止文字（`<>:"/\|?*`）と制御文字／デバイス名（`CON` `NUL` `COM1`〜`LPT9`）／先頭・末尾の空白とピリオド（`..` を含む）。サニタイズによる暗黙のリネームは行わない（元のバーコードと異なる名前で完了フォルダに入る方が業務上危険なため）。
- **同名ファイルは上書き**。完了・エラーの両フォルダで、移動前に既存ファイルを削除する。

---

## 2. 優先度別テスト対象一覧

### 【P0 — 必須】ビジネスロジック中核・データ整合性・外部連携

| 対象 | ファイル | 優先度理由 |
|---|---|---|
| `process_pdf` | `service/pdf_processor.py:109` | アプリの中核。成功／バーコード無し／例外の全分岐がファイルの移動先を決める。誤ると業務ファイルが消える |
| `is_valid_barcode` | `service/pdf_processor.py` | バーコード値をファイル名に使ってよいかの唯一の門番。ここが緩いと完了フォルダ外への書き出しやファイル消失に直結する |
| `_move_to_done_dir` | `service/pdf_processor.py:61` | バーコード値→ファイル名変換とファイル移動。R1・R2 のデータ整合性リスクが集中 |
| `_move_to_error_dir` | `service/pdf_processor.py:77` | 失敗ファイルの退避先。ここが壊れると原本が処理フォルダに残留し無限再検出の恐れ |
| `_handle_failed_file` | `service/pdf_processor.py:95` | 二重障害時の最後の砦。ファイル消失時に例外を出さず TRACE を残せるか |
| `read_barcode_from_pdf` | `service/barcode_reader.py:72` | 読み取り結果が全ての振り分けを決める。1画像の失敗で全体を止めない継続ロジックを含む |
| `_decode_code128` | `service/barcode_reader.py:56` | 2段リトライ（生画像→denoise+大津二値化）と、複数検出時の左上優先の選択規則 |
| `PDFHandler.on_created` | `service/pdf_processor.py:140` | watchdog との連携点。ディレクトリ／非PDF の除外条件を誤ると誤処理 |
| `ConfigManager.load_config` | `utils/config_manager.py:23` | 設定不在＝起動失敗。UTF-8→CP932 フォールバックは日本語Windows環境の要 |
| `AppConfig.__init__` / `save` | `utils/config_manager.py:54,77` | 全ディレクトリパスの供給源。誤読すると PDF が想定外の場所へ移動する |
| `AppConfig.ensure_directories` | `utils/config_manager.py:68` | 移動先フォルダの存在保証。未作成なら移動が全件失敗 |

**カバレッジ目標: ライン・ブランチとも 100%**

### 【P1 — 推奨】UI 影響・初期化・複雑アルゴリズム

| 対象 | ファイル | 優先度理由 |
|---|---|---|
| `extract_images_from_pdf` | `service/barcode_reader.py:25` | 埋め込み画像＋ページレンダリングの二重取得。取得順が読み取り成功率を左右（R3 のリーク検証も含む） |
| `_to_grayscale_array` | `service/barcode_reader.py:44` | 画像形状ごとの変換分岐。R9 の到達不能分岐を直接テストで押さえる |
| `setup_logging` | `utils/log_rotation.py:11` | 起動時初期化。相対パス解決・ハンドラ重複除去・不正ログレベル（R8） |
| `cleanup_old_logs` | `utils/log_rotation.py:77` | **ファイル削除を伴う**。正規表現の判定を誤ると現行ログや無関係ファイルを消す |
| `PDFProcessorApp.save_config` | `app/main_window.py:111` | ラベル値→設定への書き戻し＋ロギング再初期化。UIから設定を壊せる経路 |
| `PDFProcessorApp.process_existing_pdfs` | `app/main_window.py:130` | 起動時の一括処理。拡張子フィルタと処理順 |
| `PDFProcessorApp.start_watching` / `stop_watching` | `app/main_window.py:142,156` | Observer のライフサイクル。二重起動防止・未起動時 stop の安全性 |
| `log_trace` | `service/pdf_processor.py:39` | 運用の追跡手段。`None` の空文字化とフォーマット固定を回帰から守る |
| `get_config_value` | `utils/config_manager.py:90` | fallback の型による変換分岐（bool/int/str）。設定解釈の基盤 |

**カバレッジ目標: 90% 以上**

### 【P2 — 可能であれば実施】

| 対象 | ファイル | 優先度理由 |
|---|---|---|
| `open_error_folder` | `service/pdf_processor.py:49` | OS 分岐のみ。`os.startfile` / `subprocess.call` をモックして 3 分岐＋例外を確認 |
| `_to_enhanced_grayscale` | `service/barcode_reader.py:20` | PIL の薄いラッパー。グレースケール化とコントラスト係数の適用のみ |
| `get_config_path` | `utils/config_manager.py:7` | PyInstaller 凍結時（`sys.frozen`/`_MEIPASS`）の分岐。exe 配布の前提条件 |
| `ConfigManager.save_config` | `utils/config_manager.py:36` | 書き込みと例外ラップ（R6 のメッセージ確認を含む） |
| `get_log_info` | `utils/log_rotation.py:141` | 情報取得のみ。副作用なし |
| `setup_debug_logging` | `utils/log_rotation.py:105` | debug_mode 無効時 `None` 返却の early return が主 |
| `PDFProcessorApp.update_status` | `app/main_window.py:167` | Text ウィジェットの状態遷移（DISABLED→NORMAL→DISABLED）。モックで呼び出し順を検証 |
| `PDFProcessorApp.browse_directory` / `quit_app` / `on_closing` | `app/main_window.py:105,173,178` | ダイアログ結果に応じた分岐のみ。キャンセル時にラベルを書き換えないことだけ確認する |
| `main` | `main.py:7` | 3 行の起動配線。`setup_logging` → `Tk` → `mainloop` の順序のみ |

**カバレッジ目標: 80% 以上**

### 【P3 — テスト不要またはオプション】

| 対象 | 理由 |
|---|---|
| `utils/constants.py` 全体 | 副作用のない文字列定数。値の重複検証は保守コストのみ増える。**ただしフォーマット文字列のプレースホルダ整合（`{filename}` 等）は、それを使う側の P0/P1 テストで実質的に検証される** |
| `app/__init__.py` の `__version__` | 単一の定数定義 |
| `PDFProcessorApp.create_widgets` / `_create_directory_row` | tkinter ウィジェット配置のみ。GUI レイアウトの assert は脆く、実機目視確認が適切 |
| `PDFProcessorApp.browse_directory` | `filedialog.askdirectory` の薄いラッパー（3行） |
| `build.py` | PyInstaller 呼び出しのみ。ビルド成否は実ビルドで確認 |
| `scripts/project_structure.py` | 開発補助スクリプト。`pyproject.toml` の pyright `exclude` 対象であり製品コードではない |
| `utils/config.ini` | データファイル |

---

## 3. テスト設計方針

### 3.1 ディレクトリ構成（提案）

```
tests/
├─ conftest.py                    # 共通 fixture（下記 3.2）
├─ service/
│   ├─ test_barcode_reader.py
│   └─ test_pdf_processor.py
├─ utils/
│   ├─ test_config_manager.py
│   └─ test_log_rotation.py
└─ app/
    └─ test_main_window.py
```

### 3.2 共通 fixture

| fixture | 役割 |
|---|---|
| `app_config`（`tmp_path`） | `processing/`, `error/`, `done/` を実作成し、各属性を持つ設定オブジェクトを返す。**実ファイル移動を伴う P0 テストの土台** |
| `status_messages` | `StatusCallback` を差し替え、受信メッセージを list に蓄積するスパイ |
| `restore_root_logger`（autouse, scope=session or function） | `setup_logging` がルートロガーのハンドラを全除去・close するため、テスト前後で `logging.getLogger().handlers` を退避・復元する。**これが無いと log_rotation のテストが他テストを巻き添えにする** |
| `config_file`（`tmp_path`） | 作業フォルダを実作成し、それらを指す `config.ini` を書き出す。`app_config` の土台 |
| `pdf_in_processing` | 処理フォルダに置かれた `.pdf`（内容の妥当性は問わないケース用） |
| `patch_barcode`（テスト内ヘルパー） | `service.pdf_processor.read_barcode_from_pdf` を patch。**patch 先は定義元でなく import 先の名前空間** |

### 3.3 モック方針の原則

- **ファイルシステムはモックしない。** `tmp_path` で実操作し、移動後のファイル配置を assert する（`shutil.move` のモックでは R2 の上書きが検出できない）。
- **ネイティブ依存（pyzbar / pymupdf / cv2）はモックする。** `pyzbar.decode` は `rect.top` / `rect.left` / `data` を持つ軽量フェイクを返す。zbar DLL の有無に CI が左右されないようにする。
- **時間はモックする。** `PDFHandler.on_created` の `time.sleep(1)` は `mocker.patch("service.pdf_processor.time.sleep")` で除去。
- **tkinter はインスタンス化しない。** `PDFProcessorApp.__new__(PDFProcessorApp)` に必要属性だけ注入してメソッドを単体で呼ぶ（`__init__` が Observer 起動まで行うため）。GUI 全体はスモークテスト1本に留める。

### 3.4 主要対象のテストケース設計

#### `process_pdf`（P0）

| 種別 | ケース | 期待 |
|---|---|---|
| 正常系 | バーコード読み取り成功 | `done_dir/{barcode}.pdf` が存在、元ファイル消滅、TRACE=SUCCESS、成功メッセージが callback へ |
| 異常系 | バーコードなし（`None`） | `error_dir/{元名}.pdf` へ移動、TRACE=NO_BARCODE、警告ログ |
| 異常系 | 入力ファイル不在 | 何も移動せず早期 return、`MSG_FILE_NOT_FOUND` を callback へ |
| 異常系 | `read_barcode_from_pdf` が例外 | エラーフォルダへ退避、TRACE=ERROR、`exc_info` 付きログ |
| 異常系 | 退避自体が失敗（`shutil.move` が `PermissionError`） | 例外を外へ漏らさない、`MSG_MOVE_ERROR` を callback へ、TRACE=ERROR |
| 異常系 | `auto_open_error_folder=True/False` | エラーフォルダ自動オープンの呼び出し有無 |
| 異常系 | 不正なバーコード（`../evil`, `A/B`, `CON`, `trailing.`） | エラーフォルダへ移動、TRACE=INVALID_BARCODE（barcode 値付き）、`done_dir` に一切ファイルを作らない |
| 境界値 | バーコードが空文字 `""` | `if barcode_data:` は偽 → NO_BARCODE 経路 |
| 境界値 | `done_dir` / `error_dir` に同名ファイルが既存 | 上書きされ、移動後の内容が新しいPDFになる |
| 境界値 | 200文字ちょうど／201文字のバーコード | 200文字は完了フォルダ、201文字はエラーフォルダ |

#### `read_barcode_from_pdf` / `_decode_code128`（P0）

| 種別 | ケース | 期待 |
|---|---|---|
| 正常系 | 1画像目で検出 | その値を返し、以降の画像を処理しない（decode 呼び出し回数で確認） |
| 正常系 | 1段目失敗→二値化で検出 | `fastNlMeansDenoising` + `threshold` 経路を通り値を返す |
| 正常系 | 複数バーコード検出 | `top + left` 最小のものを採用（同値の場合の安定性も確認） |
| 異常系 | 全画像で検出できず | `None` |
| 異常系 | 途中画像で `decode` が例外 | 警告ログを出して次の画像へ継続、後続で検出できれば成功 |
| 異常系 | PDF展開自体が失敗 | 例外をそのまま送出し、`process_pdf` 側でエラーフォルダ行きとする（R4） |
| 境界値 | 画像 0 枚（画像もページも無い PDF） | `None`、例外なし |
| 境界値 | 非UTF-8バイト列の `data` | `decode('utf-8')` が `UnicodeDecodeError` → 呼び出し元の except で警告し継続 |

#### `extract_images_from_pdf`（P1）

- ページごとに「埋め込み画像すべて＋ページレンダリング1枚」の順で返ること（枚数と順序）
- 全画像がグレースケール（mode `'L'`）であること
- 複数ページ PDF での累積
- 例外時にファイルハンドルが解放されること（R3。`with` の `__exit__` 呼び出しで確認）
- 実PDFを `pymupdf` で生成して検証する（pyzbar に依存しないため CI でも実行可能）

#### `ConfigManager` / `AppConfig`（P0）

- 正常系: UTF-8 の ini 読み込み、全属性の型（`ui_width` が `int`、`auto_open_error_folder` が `bool`）
- 異常系: ファイル不在 → `FileNotFoundError`（メッセージにパスを含む）
- 異常系: CP932 で書かれた日本語 ini → フォールバックで読める
- 異常系: どちらのエンコーディングでも壊れているバイト列 → `OSError`
- 異常系: `[Directories]` セクション欠落 → `NoSectionError`
- 境界値: 任意項目の欠落 → fallback 値（`ui_width=600`, `log_dir='logs'`, `auto_open_error_folder=True`）
- 正常系: `ensure_directories` — 3つとも未作成／一部作成済み／全作成済みで戻り値のリストが変わること、既存ファイルを壊さないこと
- 正常系: `save` → 再読込で値が往復すること（ラウンドトリップ）
- 異常系: `save` 時にセクション欠落（R7）／書き込み不可（R6）

#### `cleanup_old_logs`（P1・削除を伴う）

- 保持期間を超えたローテートログのみ削除される
- 現行ログ `{project}.log` は**決して**削除されない
- 命名規則に合致しない `.log`（他アプリのログ等）は削除されない
- 境界値: 更新時刻が「ちょうど retention_days」→ `>=` により削除される
- 境界値: `retention_days=0` → 全ローテートログが削除対象
- 異常系: 個別ファイルの `os.remove` が `OSError` → ログを残して処理継続

#### `setup_logging`（P1）

- 相対パス指定時にプロジェクトルート基準へ解決される
- 2回連続実行してもハンドラが重複しない（`len(handlers) == 2`）
- 不正なログレベル文字列 → INFO へフォールバック（R8 の `TypeError` 経路も含む）
- `PermissionError` → メッセージ付きで再送出

#### `PDFHandler.on_created`（P0）

- `.pdf` → `process_pdf` が呼ばれる
- `.PDF`（大文字）→ 呼ばれる（`lower()` 判定）
- `.txt` → 呼ばれない
- `is_directory=True` → 呼ばれない
- 拡張子なし／`pdf` を含むが末尾でない名前（`report.pdf.tmp`）→ 呼ばれない

#### `app/main_window.py`（P1）

- `save_config`: ラベルのテキストが config に反映され、`save()` → `ensure_directories()` → `setup_logging()` の順で呼ばれる
- `process_existing_pdfs`: `.pdf` のみ処理、空フォルダでも例外なし、処理フォルダ不在時の挙動（現状 `os.listdir` が `FileNotFoundError`。`ensure_directories` が先行するため通常は発生しない — 順序依存の回帰テストとして価値あり）
- `start_watching`: 二重呼び出しで Observer が1つだけ生成される
- `stop_watching`: `observer is None` のとき何もしない
- スモークテスト1本: `tk` / `ttk` / `AppConfig` / `Observer` をモックし、`__init__` がタイトル・ジオメトリ・既存PDF処理・監視開始・終了ハンドラ登録まで一度通ること（ウィジェットのレイアウトそのものは検証しない）

### 3.5 実行コマンド

```bash
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short --cov=app --cov=service --cov=utils --cov-report=html
```

ブランチカバレッジを P0 の判定に使うため、`--cov-branch` の追加を推奨する。

---

## 4. テストが不要な部分の明示（理由付き）

| 対象 | 不要と判断する理由 |
|---|---|
| `utils/constants.py` の各定数 | 副作用のない文字列リテラル。値を assert するテストは定数の写経であり、変更のたびに両方を直す二重管理になる。フォーマット文字列の妥当性は利用側テストで担保される |
| `app/__init__.py:__version__` | 単一の定数。バージョン更新のたびにテストが落ちるだけで欠陥を検出しない |
| `create_widgets` / `_create_directory_row` | tkinter のウィジェット生成・grid 配置。振る舞いではなくレイアウト記述であり、モック上での assert は実際の表示崩れを検出できない。実機での目視確認が適切な検証手段（起動時に例外を出さないことのみスモークテストで担保） |
| `build.py` | `subprocess.run` に固定引数を渡すのみ。引数リストの assert は PyInstaller の実挙動を保証しない。exe の起動確認で代替する |
| `scripts/project_structure.py` | 製品コードではない開発補助ツール（pyright の `exclude` 対象）。壊れても本番機能に影響しない |
| `main.py` の `if __name__ == "__main__"` ブロック | エントリポイントの慣用句。`main()` 本体は P2 でテスト済み |
| pyzbar / pymupdf / OpenCV 自体の挙動 | サードパーティの責務。CODE128 の実デコード精度は Windows 実機での統合テスト（実PDFサンプル数点）で確認し、ユニットテストの範囲外とする |

---

## 5. 実装結果

P0〜P2 を実装済み（159件）。テストファイル構成は 3.1 のとおり。

| ファイル | 主対象 | 件数 |
|---|---|---|
| `tests/conftest.py` | 共通fixture | — |
| `tests/service/test_pdf_processor.py` | P0: `process_pdf` / `is_valid_barcode` / `PDFHandler`、P2: `open_error_folder` | 56 |
| `tests/service/test_barcode_reader.py` | P0: `read_barcode_from_pdf` / `_decode_code128`、P1: `extract_images_from_pdf` | 20 |
| `tests/utils/test_config_manager.py` | P0: `ConfigManager` / `AppConfig`、P1: `get_config_value` | 29 |
| `tests/utils/test_log_rotation.py` | P1: `setup_logging` / `cleanup_old_logs`、P2: `setup_debug_logging` / `get_log_info` | 33 |
| `tests/app/test_main_window.py` | P1: `save_config` / 監視制御 / 既存PDF処理、P2: `update_status` ほか | 19 |
| `tests/test_main.py` | P2: 起動処理の配線 | 2 |

### カバレッジ（`--cov-branch`）

| モジュール | 目標 | 実績 |
|---|---|---|
| `service/pdf_processor.py` | P0: 100% | **100%** |
| `service/barcode_reader.py` | P0: 100% | **100%** |
| `utils/config_manager.py` | P0: 100% | **100%** |
| `utils/log_rotation.py` | P1: 90%+ | **100%** |
| `app/main_window.py` | P1: 90%+ | **100%** |
| `main.py` | P2: 80%+ | 83%（未達分は `if __name__ == "__main__"` の1行のみ） |

全体 99%（511ステートメント / 104ブランチ）。`pyright`（standard, tests含む）はエラー0。

### 残る検証

pyzbar の実デコードはモックしているため、CODE128 の読み取り精度は Windows 実機で実PDFを用いて確認する（`.claude/rules` および CLAUDE.md の方針どおり）。
