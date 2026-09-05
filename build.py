import subprocess


def build_executable() -> None:
    subprocess.run([
        "pyinstaller",
        "--name=BarcodePDF",
        "--windowed",
        "--icon=assets/BarcodePDF.ico",
        "--add-data", "utils/config.ini:.",
        "--collect-binaries", "pyzbar",
        "main.py"
    ], check=True)

    print("Executable built successfully.")


if __name__ == "__main__":
    build_executable()
