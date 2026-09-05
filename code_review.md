# コードレビュー: BarcodePDF

対象: `main.py` / `build.py` / `app/` / `service/` / `utils/`（計 904 行）
観点: **可読性・メンテナンス性の向上**、**KISS の原則**
前提: レビュー時点で `pytest tests/ -q` は **168 passed**（全緑）

---

## 総評

責務分割（`app` = GUI / `service` = 処理 / `utils` = 基盤）は明快で、`constants.py` によるメッセージ一元化、`PdfWatcher` の「サイズ+更新日時が安定してから処理する」設計、`update_status` のキュー経由でのスレッド安全化など、要点は正しく押さえられています。コメントも「なぜそうしたか」を書けており良質です。

一方で **`utils/` 層に不要な複雑さが集中** しています。特に `get_config_value` が `object` を返す設計が `log_rotation.py` 全体にボイラープレートと `# type: ignore` を撒いており、ここが本コードベース最大の可読性負債です。また `service/` には再起動不能や無限リトライといった、レビューで見つけるべきクラスの不具合が 2 件あります。

修正の投資対効果は以下の順です。

| 優先 | 項目 | 効果 |
|---|---|---|
| P0 | 1. `PdfWatcher` の再起動不能 | バグ（検証済み） |
| P0 | 2. 移動失敗ファイルの無限リトライ | バグ（検証済み） |
| P1 | 3. `get_config_value` の廃止 | **-25 行**・`# type: ignore` 2 件解消 |
| P1 | 4. ログ/PDF 削除ロジックの重複統合 | **-30 行**・規約違反解消 |
| P1 | 5. `barcode_reader` の import 時 config 読み込み | 設計・テスト容易性 |
| P2 | 6〜11. 局所的な単純化・規約準拠 | 可読性 |

---

## P0-1. `PdfWatcher` は stop 後に start しても動かない

`service/pdf_watcher.py:58-64`

`stop()` が `_stop_event` を **clear していない** ため、再度 `start()` してもスレッドは `_run` の `wait()` で即座に True を受け取り、1 回も走査せず終了します。

検証結果:

```
w.start(); w.stop(); w.start()
→ restart後もスレッド生存: False
```

現状 UI からは終了時にしか `stop()` を呼ばないため顕在化していませんが、「設定保存時に監視を貼り直す」という自然な拡張（項目 5 と関連）を入れた瞬間に **監視が黙って止まる** 最悪の壊れ方をします。1 行で直せるので今直すべきです。

```diff
--- a/service/pdf_watcher.py
+++ b/service/pdf_watcher.py
@@ def start(self) -> None:
     def start(self) -> None:
         if self._thread:
             return

+        self._stop_event.clear()
         self._thread = threading.Thread(target=self._run, daemon=True)
         self._thread.start()
```

---

## P0-2. 移動に失敗したファイルが 2 スキャンごとに再処理され続ける

`service/pdf_watcher.py:70-86`

`scan_once` は処理したファイルを `stable_signatures` に入れずに `continue` します。処理でファイルが移動されれば消えるので通常は問題ありませんが、**移動に失敗してファイルが取込フォルダに残った場合**、署名が登録され直し → 次スキャンで再処理、というループになります。

```
scan1: 署名を登録
scan2: 署名一致 → process_pdf（移動失敗、ファイルは残る）→ 署名を破棄
scan3: 署名を再登録
scan4: 再び process_pdf …（4 秒周期で永久に繰り返す）
```

`process_pdf` は広い `except` で `_handle_failed_file` に流し、そこも失敗すればファイルは残ります。スキャナーがファイルをロックしている、エラーフォルダの権限がない、といった現実的なケースで **PDF レンダリング + バーコードデコードを 4 秒おきに永久実行し、ログを埋め尽くします**。

処理を試みたファイルを記録し、「署名が変わらない限り再処理しない」ようにするのが最小の修正です。`__init__` に `self._handled: set[str] = set()` を追加した上で:

```diff
     def scan_once(self) -> None:
-        """target_dir を1回走査し、前回と同じ状態のPDFを処理する"""
+        """target_dir を1回走査し、前回と同じ状態のPDFを処理する
+
+        処理済みファイルの署名も保持する。移動に失敗して残ったファイルを、
+        内容が変わらないまま繰り返し処理しないため。
+        """
         stable_signatures: dict[str, FileSignature] = {}

         for entry in self._scan_pdf_entries():
             signature = _file_signature(entry.path)
             if signature is None:
                 continue

-            # 前回と同じサイズ・更新日時なら書き込みが完了している
-            if self._signatures.get(entry.path) == signature:
-                process_pdf(entry.path, self.config, self.status_callback)
-                continue
-
-            stable_signatures[entry.path] = signature
+            previous = self._signatures.get(entry.path)
+            stable_signatures[entry.path] = signature
+
+            # 前回と同じサイズ・更新日時になった初回だけ処理する
+            if previous == signature and entry.path not in self._handled:
+                self._handled.add(entry.path)
+                process_pdf(entry.path, self.config, self.status_callback)

         self._signatures = stable_signatures
+        # 消えたファイルの記録は破棄する（同名で再投入されたら処理対象に戻す）
+        self._handled &= stable_signatures.keys()
```

実装方法は他にもありますが、要点は **「同じ内容のファイルを二度処理しない」という不変条件をコードに明示すること** です。

---

## P1-3. `get_config_value` を廃止し、configparser の型付きアクセサに寄せる

`utils/config_manager.py:93-107` と `utils/log_rotation.py` 全体

`get_config_value` は戻り値が `object` のため、呼び出し側が必ず `str(...)` / `int(...)` でキャストし直す必要があります。その結果 `setup_logging` の冒頭は「値を取る 4 行」と「None ガードしてキャストする 4 行」に分裂し、`# type: ignore` が 2 箇所（`log_rotation.py:22`, `:193`）発生しています。

```python
# 現状: 9 行 + type: ignore
log_directory_value = get_config_value(config, 'LOGGING', 'log_directory', 'logs')
log_retention_days_value = get_config_value(config, 'LOGGING', 'log_retention_days', 7)
project_name_value = get_config_value(config, 'LOGGING', 'project_name', 'BarcodePDF')
log_level_value = get_config_value(config, 'LOGGING', 'log_level', 'INFO')

log_directory = str(log_directory_value if log_directory_value is not None else 'logs')
log_retention_days = int(log_retention_days_value if log_retention_days_value is not None else 7)  # type: ignore
project_name = str(project_name_value if project_name_value is not None else 'BarcodePDF')
log_level = str(log_level_value if log_level_value is not None else 'INFO')
```

`configparser` は `get(fallback=)` / `getint(fallback=)` / `getboolean(fallback=)` を標準で備えており、**同じ意味論（セクション・キー欠落時に fallback）を型付きで**提供します。実際 `AppConfig` は既にそちらを使っており、**同じプロジェクト内に設定アクセス手段が 2 系統併存している**状態です。

```diff
--- a/utils/log_rotation.py
+++ b/utils/log_rotation.py
@@
-        log_directory_value = get_config_value(config, 'LOGGING', 'log_directory', 'logs')
-        log_retention_days_value = get_config_value(config, 'LOGGING', 'log_retention_days', 7)
-        project_name_value = get_config_value(config, 'LOGGING', 'project_name', 'BarcodePDF')
-        log_level_value = get_config_value(config, 'LOGGING', 'log_level', 'INFO')
-
-        log_directory = str(log_directory_value if log_directory_value is not None else 'logs')
-        log_retention_days = int(log_retention_days_value if log_retention_days_value is not None else 7)  # type: ignore
-        project_name = str(project_name_value if project_name_value is not None else 'BarcodePDF')
-        log_level = str(log_level_value if log_level_value is not None else 'INFO')
+        log_directory = config.get('LOGGING', 'log_directory', fallback='logs')
+        log_retention_days = config.getint('LOGGING', 'log_retention_days', fallback=7)
+        project_name = config.get('LOGGING', 'project_name', fallback='BarcodePDF')
+        log_level = config.get('LOGGING', 'log_level', fallback='INFO')
```

`setup_debug_logging`（`:146`, `:151`）、`get_log_info`（`:182-194`）、`cleanup_error_pdfs`（`:109`）も同様に置換でき、**合計 25 行前後と `# type: ignore` 2 件が消えます**。置換後 `get_config_value` は無参照になるので、関数本体（15 行）と `tests/utils/test_config_manager.py:222-264` のテスト群も併せて削除してください。

注: `getboolean` は `'on'` も真として解釈する点だけ現行と挙動が異なります（現行は `true/1/yes` のみ）。設定ファイル向けとしては標準の解釈のほうが妥当で、実害はありません。

さらに `log_directory` の相対パス解決が **3 箇所で同一コピー**（`:26-28`, `:153-155`, `:184-186`）されています。上記と併せて小関数に括り出すのが自然です。

```python
def _resolve_log_directory(config: configparser.ConfigParser) -> str:
    """LOGGING/log_directory を絶対パスで返す（相対指定はプロジェクトルート基準）"""
    directory = config.get('LOGGING', 'log_directory', fallback='logs')
    if os.path.isabs(directory):
        return directory
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)
```

---

## P1-4. 「保存期間を過ぎたファイルを削除する」ロジックが 2 重化している

`utils/log_rotation.py:79-104`（`cleanup_old_logs`）と `:107-138`（`cleanup_error_pdfs`）

両者は「対象ファイルを絞り込む条件」だけが違い、残りの **走査 → 更新日時取得 → 保持期間比較 → 削除 → 件数ログ → 例外握り潰し** が完全に同型です。`.claude/rules/python-coding.md` の「類似のロジックが 2 箇所に存在する場合は共有関数にリファクタリングする」に正面から抵触します。

```python
def _delete_files_older_than(
    directory: str,
    retention_days: int,
    is_target: Callable[[str], bool],
    label: str,
) -> None:
    """directory 内の対象ファイルのうち、保存期間を過ぎたものを削除する"""
    if not os.path.isdir(directory):
        return

    threshold = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not is_target(filename) or not os.path.isfile(file_path):
            continue
        try:
            if datetime.fromtimestamp(os.path.getmtime(file_path)) <= threshold:
                os.remove(file_path)
                deleted_count += 1
        except OSError as e:
            logging.error(MSG_DELETE_FAILED.format(label=label, filename=filename, error=e))

    if deleted_count > 0:
        logging.info(MSG_DELETE_SUMMARY.format(count=deleted_count, label=label))
```

呼び出し側は各 25〜30 行が 3〜4 行になります。

```python
def cleanup_old_logs(log_directory: str, retention_days: int, project_name: str) -> None:
    pattern = re.compile(rf'{re.escape(project_name)}\.log\.\d{{4}}-\d{{2}}-\d{{2}}\.log$')
    _delete_files_older_than(
        log_directory, retention_days, lambda name: bool(pattern.match(name)), LABEL_LOG_FILE
    )


def cleanup_error_pdfs(config: configparser.ConfigParser, retention_days: int) -> None:
    error_directory = config.get('Directories', 'error_dir', fallback='')
    _delete_files_older_than(
        error_directory, retention_days, lambda name: name.lower().endswith('.pdf'), LABEL_ERROR_PDF
    )
```

**併せて指摘: 責務の置き場所が違います。** `cleanup_error_pdfs` は PDF の後始末であってログ機能ではありません。現在は「ログ設定の初期化ついでに実行される」ため、`log_rotation.setup_logging` を呼ぶと副作用で業務データ（エラー PDF）が消えます。`setup_logging` の呼び出し箇所（`main.py:8`, `app/main_window.py:118`）を読んだだけでは PDF が削除されることに気づけません。`service/` 側へ移し、`main.py` から明示的に呼ぶ形が読み手に親切です。

---

## P1-5. `barcode_reader` の設定値が import 時に固定される

`service/barcode_reader.py:17-28`

```python
_config = load_config()
CONTRAST_FACTOR = _config.getfloat('Barcode', 'contrast_factor', fallback=2.0)
RENDER_ZOOM = _config.getfloat('Barcode', 'render_zoom', fallback=2.0)
...
```

モジュール import 時に副作用でファイル I/O が走る設計には 3 つの問題があります。

1. **`config.ini` が無いと ImportError になる。** `ConfigManager.load_config` は `FileNotFoundError` を送出するため、設定不備が「バーコード読み取りモジュールが import できない」という無関係な症状で現れます。
2. **テストで差し替えられない。** 定数なので pytest から倍率を変えた検証ができません（他モジュールが `AppConfig` を注入可能なのと非対称です）。
3. **GUI で設定を保存しても反映されない。** アプリ再起動が必要ですが、そのことがコードのどこにも書かれていません。

他モジュールと同様、`AppConfig` に読ませて引数で渡すのが一貫します。

```diff
--- a/utils/config_manager.py
+++ b/utils/config_manager.py
@@ class AppConfig:
         self.auto_open_error_folder: bool = self.config.getboolean(
             'Options', 'auto_open_error_folder', fallback=True
         )
+        self.contrast_factor: float = self.config.getfloat('Barcode', 'contrast_factor', fallback=2.0)
+        # 72dpi基準の拡大率。等倍ではバーの太さが足りずデコードできない
+        self.render_zoom: float = self.config.getfloat('Barcode', 'render_zoom', fallback=2.0)
+        # ページ上端から探索する高さの割合
+        self.top_band_ratio: float = self.config.getfloat('Barcode', 'top_band_ratio', fallback=0.15)
+        self.min_barcode_width_ratio: float = self.config.getfloat(
+            'Barcode', 'min_barcode_width_ratio', fallback=0.20
+        )
```

`read_barcode_from_pdf(pdf_path: str, config: AppConfig) -> str | None` とし、`process_pdf` は既に `config` を持っているのでそのまま渡せます。**呼び出し側の変更は 1 行**です。

なお、設定を「保存しても反映されない」点は `render_zoom` に限らず `target_dir` も同様です（`PdfWatcher` は `AppConfig` インスタンスを共有しているので `target_dir` は偶然反映されますが、これは暗黙の依存です）。`save_config` で監視を貼り直すのが明示的で安全です（P0-1 の修正が前提）。

```diff
--- a/app/main_window.py
+++ b/app/main_window.py
@@ def save_config(self) -> None:
         self.config.save()
         self.ensure_directories()

+        # 新しい取込フォルダを対象にするため監視を貼り直す
+        self.stop_watching()
+        self.start_watching()
+
         setup_logging(self.config.config)
```

---

## P2-6. `process_pdf` の 3 分岐を平坦化する

`service/pdf_processor.py:134-166`

3 つの分岐が「メッセージ組み立て → `logger.warning` → `status_callback` → `_move_to_error_dir`」という同じ 4 手順を繰り返しており、`if/elif/else` の各ブロックが縦に伸びて主眼（成功／失敗の分岐）が読み取りにくくなっています。「失敗理由」を 1 組の値にまとめると、重複が消えます。

```diff
-        if barcode_data and is_valid_barcode(barcode_data):
-            _move_to_done_dir(pdf_path, barcode_data, config, status_callback)
-        elif barcode_data:
-            message = MSG_BARCODE_INVALID.format(
-                filename=os.path.basename(pdf_path), barcode=barcode_data
-            )
-            logger.warning(message)
-            status_callback(message)
-            _move_to_error_dir(
-                pdf_path, config, status_callback, TRACE_RESULT_INVALID_BARCODE, barcode_data
-            )
-        else:
-            message = MSG_BARCODE_NOT_FOUND.format(filename=os.path.basename(pdf_path))
-            logger.warning(message)
-            status_callback(message)
-            _move_to_error_dir(pdf_path, config, status_callback, TRACE_RESULT_NO_BARCODE)
+        filename = os.path.basename(pdf_path)
+
+        if barcode_data and is_valid_barcode(barcode_data):
+            _move_to_done_dir(pdf_path, barcode_data, config, status_callback)
+            return
+
+        if barcode_data:
+            result = TRACE_RESULT_INVALID_BARCODE
+            message = MSG_BARCODE_INVALID.format(filename=filename, barcode=barcode_data)
+        else:
+            result = TRACE_RESULT_NO_BARCODE
+            message = MSG_BARCODE_NOT_FOUND.format(filename=filename)
+
+        logger.warning(message)
+        status_callback(message)
+        _move_to_error_dir(pdf_path, config, status_callback, result, barcode_data)
```

## P2-7. 「上書き移動」が 2 箇所にコピーされている

`service/pdf_processor.py:86-89` と `:106-109` に、コメントを含めて同じ 4 行があります。

```python
def _move_overwriting(source: str, destination: str) -> None:
    """同名ファイルがあれば上書きして移動する"""
    if os.path.exists(destination):
        os.remove(destination)
    shutil.move(source, destination)
```

また両関数とも `shutil.move` の **後** に `os.path.basename(pdf_path)` を呼んでいます（`:91`, `:111`）。文字列操作なので動作しますが、「移動済みのパスをまだ参照している」ように読めます。移動前に `filename` を取っておくと意図が明確になります。

## P2-8. `except Exception` で握り潰し・再送出している箇所

`utils/log_rotation.py:75-76` の `except Exception as e: raise Exception(f"...: {e}")` は、**元の例外型とトレースバックを捨てて** 汎用 `Exception` に変換しています。呼び出し側（`main.py`）は型で判別できず、原因究明も難しくなります。`raise` を素通しにするか、少なくとも `from e` を付けてください。直上の `PermissionError` 再送出（`:73-74`）も同様に `from e` が必要です。

`cleanup_old_logs` / `cleanup_error_pdfs` / `setup_debug_logging` / `get_log_info` の広い `except Exception` も、想定される失敗（`OSError`）は内側で個別に捕捉済みです。外側の網は「あり得ないシナリオに対するエラーハンドリング」に当たり、不具合を静かに隠します。

## P2-9. プロダクションコードから未使用の関数

- `utils/log_rotation.py:141` `setup_debug_logging` — 呼び出し元はテストのみ。`config.ini` に `debug_mode = True` が設定されているのに **誰も呼んでいない**ため、デバッグログは実際には出力されません。設定と実装が乖離しています。
- `utils/log_rotation.py:177` `get_log_info` — 呼び出し元はテストのみ（31 行）。
- `utils/config_manager.py:43` `ConfigManager.get_path` — 呼び出し元はテストのみ。参照する `Paths` セクションは `config.ini` に存在しません。
- `utils/config_manager.py:66` `AppConfig.start_minimized` — 読み込むだけで、ウィンドウの最小化に使われていません。

いずれも「テストがあるので緑」ですが、テストが仕様ではなく実装を追認しているだけの状態です。方針決定はお任せしますが（規約に従い、指示なく削除はしません）、**`setup_debug_logging` は「呼ぶ」か「消す」かの二択**で、現状の宙ぶらりんが一番コストが高いです。

## P2-10. `is_valid_barcode` のデバイス名判定が拡張子付きを取りこぼす

`service/pdf_processor.py:49`

Windows のデバイス名予約は拡張子の有無に関わらず適用されるため、`CON.foo` というバーコードは検証を通過し、`CON.foo.pdf` の作成で失敗します。最初のピリオドまでで判定するのが正確です。

```diff
-    if barcode.upper() in RESERVED_FILENAMES:
+    if barcode.upper().split('.')[0] in RESERVED_FILENAMES:
         return False
```

発生確率は低いため、優先度は低で構いません。

## P2-11. 規約準拠の細かい点

- **`utils/config_manager.py`**: トップレベル定義の前後が 1 行空行になっています（`:6-7`, `:14-15`, `:16-17`）。PEP8 は 2 行を要求します。
- **`build.py:14`**: `print(f"Executable built successfully.")` — プレースホルダが無く `f` が不要です。また `build_executable()` に戻り値型ヒントがなく（規約は型ヒント必須）、`subprocess.run` の `returncode` を確認していないため、**PyInstaller が失敗しても「成功しました」と表示されます**。`check=True` を付けてください。
- **`app/main_window.py:118`**: `setup_logging(self.config.config)` の `config.config` という二重表記は、`AppConfig` が内部の `ConfigParser` を公開していることの表れです。`AppConfig.setup_logging()` のようなメソッドを生やすか、`log_rotation` 側が `AppConfig` を受け取る形にすると呼び出しが読みやすくなります。
- **`utils/log_rotation.py`**: ログメッセージが f-string 直書きで、`constants.py` の一元管理方針から外れています。UI 表示メッセージではないため規約の直接の対象外とも読めますが、他モジュールとの一貫性の観点で整理を検討してください。

---

## 良かった点

- `PdfWatcher` の docstring（`:1-6`）が「なぜイベント監視ではなくポーリングなのか」を説明しており、将来の「watchdog に置き換えよう」という誤った改善提案を確実に防いでいます。**このコードベースで最も価値の高いコメント**です。
- `update_status` のキュー経由（`app/main_window.py:141-153`）による tkinter のスレッド安全性確保。docstring で理由まで書かれています。
- `_render_top_band` の `with pymupdf.open(...)`（`:33-34`）— 破損 PDF でのハンドルリークを意識した、コメント付きの対処。
- `is_valid_barcode` の `barcode.strip(' .') == barcode`（`:51-52`）— Windows の前後空白・ピリオド除去という非自明な仕様を 1 行で表現し、`'..'` も同時に弾いている点が簡潔です。
- `TRACE_FORMAT` による構造化トレースログ。障害調査で効きます。
- 168 件のテストが 1.16 秒で完走し、P0/P1 のリファクタリングを安全に実施できる土台があります。

---

## 推奨する適用順序

各ステップで `.venv\Scripts\python.exe -m pytest tests/ -v --tb=short` が緑であることを確認してください。

1. **P0-1, P0-2**（`pdf_watcher.py`）→ 検証: 再起動テストと「移動失敗時に再処理されない」テストを追加し、パスさせる
2. **P1-3**（`get_config_value` 廃止）→ 検証: 既存テスト全緑 + pyright で `# type: ignore` が不要になったことを確認
3. **P1-4**（削除ロジック統合 + `cleanup_error_pdfs` の移設）→ 検証: `test_log_rotation.py` 全緑
4. **P1-5**（`barcode_reader` の設定注入）→ 検証: `test_barcode_reader.py` 全緑、Windows 実機でバーコード読み取り確認（pyzbar はネイティブ DLL 依存のため必須）
5. **P2 群** → 検証: 全緑

P0 の 2 件のみでも独立して適用可能です。
