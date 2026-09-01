import subprocess


def build_executable():
    subprocess.run([
        "pyinstaller",
        "--name=app_name",
        "--windowed",
        "--add-data", "utils/config.ini:.",
        "main.py"
    ])

    print(f"Executable built successfully. Version: {new_version}")


if __name__ == "__main__":
    build_executable()
