import configparser
import os
import sys
from pathlib import Path
from typing import Final


def get_config_path() -> Path:
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    return base_path / 'config.ini'


CONFIG_PATH: Final[Path] = get_config_path()


class ConfigManager:
    def __init__(self, config_file: Path | str = CONFIG_PATH) -> None:
        self.config_file: Path = Path(config_file)
        self.config: configparser.ConfigParser = configparser.ConfigParser()
        self.load_config()

    def load_config(self) -> None:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        try:
            self.config.read(self.config_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content: str = self.config_file.read_bytes().decode('cp932')
                self.config.read_string(content)
            except (UnicodeDecodeError, OSError) as e:
                raise OSError(f"Failed to load config: {e}") from e

    def save_config(self) -> None:
        try:
            with open(self.config_file, 'w', encoding='utf-8') as configfile:
                self.config.write(configfile)
        except (IOError, OSError) as e:
            raise OSError(f"Failed to save config: {e}") from e

    def ensure_section(self, section: str) -> None:
        if section not in self.config:
            self.config[section] = {}


class AppConfig:
    """config.ini のアプリ設定を型付きで読み書きする"""

    def __init__(self, config_file: Path | str = CONFIG_PATH) -> None:
        self._manager: ConfigManager = ConfigManager(config_file)
        self.config: configparser.ConfigParser = self._manager.config
        self.target_dir: str = self.config.get('Directories', 'target_dir')
        self.error_dir: str = self.config.get('Directories', 'error_dir')
        self.done_dir: str = self.config.get('Directories', 'done_dir')
        self.log_dir: str = self.config.get('LOGGING', 'log_directory', fallback='logs')
        self.log_retention_days: int = self.config.getint('LOGGING', 'log_retention_days', fallback=7)
        self.ui_width: int = self.config.getint('UI', 'width', fallback=600)
        self.ui_height: int = self.config.getint('UI', 'height', fallback=500)
        # ステータスキューを取り出す間隔（ミリ秒）
        self.status_poll_ms: int = self.config.getint('UI', 'status_poll_ms', fallback=200)
        self.auto_open_error_folder: bool = self.config.getboolean(
            'Options', 'auto_open_error_folder', fallback=True
        )
        self.contrast_factor: float = self.config.getfloat('Barcode', 'contrast_factor', fallback=2.0)
        self.render_zoom: float = self.config.getfloat('Barcode', 'render_zoom', fallback=2.0)
        # ページ上端から探索する高さの割合
        self.top_band_ratio: float = self.config.getfloat('Barcode', 'top_band_ratio', fallback=0.15)
        # ページ幅に対する最小幅。帯の中に小さなバーコードが並んでいても大きい方だけを採用する
        self.min_barcode_width_ratio: float = self.config.getfloat(
            'Barcode', 'min_barcode_width_ratio', fallback=0.20
        )

    def ensure_directories(self) -> list[str]:
        """取込・エラー・完了フォルダを作成し、新規作成したパスを返す"""
        created: list[str] = []
        for directory in (self.target_dir, self.error_dir, self.done_dir):
            if not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
                created.append(directory)
        return created

    def save(self) -> None:
        for section in ('Directories', 'LOGGING', 'Options'):
            self._manager.ensure_section(section)

        self.config['Directories']['target_dir'] = self.target_dir
        self.config['Directories']['error_dir'] = self.error_dir
        self.config['Directories']['done_dir'] = self.done_dir
        self.config['LOGGING']['log_directory'] = self.log_dir
        self.config['Options']['auto_open_error_folder'] = str(self.auto_open_error_folder)
        self._manager.save_config()


def load_config() -> configparser.ConfigParser:
    return ConfigManager().config

