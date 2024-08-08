import subprocess
import shutil
from version_manager import update_version, update_main_py


def build_executable():
    new_version = update_version()
    update_main_py(new_version)

    subprocess.run([
        "pyinstaller",
        "BarcodePDF.spec",
    ])

    # 必要なファイルをdistフォルダにコピー
    shutil.copy("config.ini", "dist/BarcodePDF/")
    shutil.copy("version.txt", "dist/BarcodePDF/")

    print(f"Executable built successfully. Version: {new_version}")


if __name__ == "__main__":
    build_executable()
