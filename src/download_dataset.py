import os
import shutil
import kagglehub

def download_dataset():
    kagglehub.login()

    path = kagglehub.dataset_download("frankiscoo/all-subset")
    dest_folder = "data/raw"

    os.makedirs(dest_folder, exist_ok=True)

    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(dest_folder, item)
        shutil.move(s, d)

    print(f"Dataset installed at {dest_folder}")
