import os
import shutil
import configparser


def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['Directories']['source_directory'], config['Directories']['destination_directory']


def move_files(source_dir, destination_dir):
    # ソースディレクトリとデスティネーションディレクトリが存在することを確認
    if not os.path.exists(source_dir):
        print(f"エラー: ソースディレクトリ {source_dir} が見つかりません。")
        return
    if not os.path.exists(destination_dir):
        print(f"デスティネーションディレクトリ {destination_dir} が存在しません。作成します。")
        os.makedirs(destination_dir)

    # ソースディレクトリ内のすべてのファイルを取得
    files = os.listdir(source_dir)

    # ファイルを移動
    for file in files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)

        try:
            shutil.move(source_path, destination_path)
            print(f"{file} を移動しました。")
        except Exception as e:
            print(f"{file} の移動中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    try:
        source_directory, destination_directory = load_config()
        move_files(source_directory, destination_directory)
    except Exception as e:
        print(f"設定ファイルの読み込み中にエラーが発生しました: {str(e)}")
        print("config.iniファイルが正しく設定されていることを確認してください。")
