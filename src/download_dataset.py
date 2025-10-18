import os
import shutil
import kagglehub
import os

kagglehub.login()

path = kagglehub.dataset_download("frankiscoo/all-subset")
print("Path to downloaded file:", path)

dest_folder = "data/raw/ALL"
nested_folder = os.path.join(path, "datasets/frankiscoo/all-subset/versions/1")

# Move all files from nested_folder to dest_folder
for item in os.listdir(nested_folder):
    s = os.path.join(nested_folder, item)
    d = os.path.join(dest_folder, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)

shutil.rmtree(os.path.join(path, "datasets"))

print(f"Dataset installed at {dest_folder}")
