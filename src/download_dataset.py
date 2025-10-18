import kagglehub

kagglehub.login()

path = kagglehub.dataset_download("obulisainaren/multi-cancer", path="Multi Cancer/Multi Cancer/ALL")
print("Path to download:", path)