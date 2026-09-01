import subprocess


def build_executable():
    subprocess.run([
        "pyinstaller",
        "--name=BarcodePDF",
        "--windowed",
        "--icon=assets/BarcodePDF.ico",
        "--add-data", "utils/config.ini:.",
        "--collect-binaries", "pyzbar",
        "main.py"
    ])

    print(f"Executable built successfully.")


if __name__ == "__main__":
    build_executable()
