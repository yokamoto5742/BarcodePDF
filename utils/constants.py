"""UI表示とログ出力に使用するメッセージ定数"""

APP_TITLE = "BarcodePDF"

LABEL_TARGET_DIR = "取込フォルダ:"
LABEL_ERROR_DIR = "エラーフォルダ:"
LABEL_DONE_DIR = "完了フォルダ:"
LABEL_LOG_DIR = "ログフォルダ:"
LABEL_STATUS = "ステータス:"

BUTTON_BROWSE = "参照"
BUTTON_SAVE_CONFIG = "設定を保存"
BUTTON_CLOSE = "閉じる"
CHECKBOX_AUTO_OPEN_ERROR_FOLDER = "エラーフォルダを自動的に開く"

DIALOG_SAVE_CONFIG_TITLE = "設定保存"
DIALOG_SAVE_CONFIG_MESSAGE = "設定が保存されました。"
DIALOG_QUIT_TITLE = "終了"
DIALOG_QUIT_MESSAGE = "アプリケーションを終了しますか？"

MSG_CONFIG_UPDATED = "設定が更新されました"
MSG_DIRECTORY_CREATED = "フォルダを作成しました: {directory}"
MSG_WATCH_STARTED = "{directory} の監視を開始しました..."
MSG_WATCH_STOPPED = "監視を停止しました。"
MSG_APP_QUIT = "アプリケーションを終了します"

MSG_SCAN_ERROR = "{directory} の走査中にエラーが発生しました: {error}"

MSG_PROCESSING_START = "PDFの処理を開始: {path}"
MSG_FILE_NOT_FOUND = "ファイルが見つかりません: {path}"
MSG_PROCESS_DONE = "処理完了: {source} -> {destination}"
MSG_BARCODE_NOT_FOUND = "{filename} からバーコードが見つかりませんでした"
MSG_BARCODE_INVALID = "{filename} のバーコードはファイル名に使用できません: {barcode}"
MSG_MOVED_TO_ERROR = "{filename} をエラーフォルダーに移動しました"
MSG_PROCESS_ERROR = "{filename} の処理中にエラーが発生しました: {error}"
MSG_MOVE_ERROR = "ファイルの移動中にエラーが発生しました: {error}"
MSG_BARCODE_READ_ERROR = "バーコード読み取り中にエラーが発生しました: {error}"
MSG_OPEN_ERROR_FOLDER_FAILED = "エラーフォルダを開く際にエラーが発生しました: {error}"
MSG_OPEN_ERROR_FOLDER_UNSUPPORTED = "エラーフォルダを開けません: {path}"

# ファイルの移動元・移動先を追跡するための構造化ログ行
TRACE_FORMAT = "TRACE result={result} src={source} barcode={barcode} dst={destination}"
TRACE_RESULT_SUCCESS = "SUCCESS"
TRACE_RESULT_NO_BARCODE = "NO_BARCODE"
TRACE_RESULT_INVALID_BARCODE = "INVALID_BARCODE"
TRACE_RESULT_ERROR = "ERROR"
