import os
import shutil


def move_files(source_dir, destination_dir):
    if not os.path.exists(source_dir):
        print(f"エラー: {source_dir} が見つかりません。")
        return
    if not os.path.exists(destination_dir):
        print(f" {destination_dir} が存在しません。作成します。")
        os.makedirs(destination_dir)

    files = os.listdir(source_dir)

    for file in files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)

        try:
            shutil.move(source_path, destination_path)
            print(f"{file} を移動しました。")
        except Exception as e:
            print(f"{file} の移動中にエラーが発生しました: {str(e)}")


source_directory = r"C:\Shinseikai\BarcodePDF\preprocessing"
destination_directory = r"C:\Shinseikai\BarcodePDF\processing"

move_files(source_directory, destination_directory)
