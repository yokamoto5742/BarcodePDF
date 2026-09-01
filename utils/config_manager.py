import configparser
import os
import sys
from pathlib import Path
from typing import Final

def get_config_path() -> Path:
    # 実行ファイルのディレクトリを取得
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
            raise OSError(f"Failed to load config: {e}") from e

    def get_path(self, key: str) -> Path:
        return Path(self.config.get('Paths', key))

    def _ensure_section(self, section: str) -> None:
        if section not in self.config:
            self.config[section] = {}


def load_config() -> configparser.ConfigParser:
    return ConfigManager().config


def get_config_value(
    config: configparser.ConfigParser,
    section: str,
    key: str,
    fallback: object = None,
) -> object:
    try:
        value = config.get(section, key)
        if isinstance(fallback, bool):
            return value.lower() in ('true', '1', 'yes')
        if isinstance(fallback, int):
            return int(value)
        return value
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback
