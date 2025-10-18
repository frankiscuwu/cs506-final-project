import os
import shutil
import kagglehub
import os

kagglehub.login()

path = kagglehub.dataset_download("frankiscoo/all-subset")
dest_folder = "data/raw"

# Move all files from nested_folder to dest_folder
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest_folder, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)

shutil.rmtree("data/raw/ALL/datasets")

print(f"Dataset installed at {dest_folder}")
