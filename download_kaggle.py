import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "netflix-inc/netflix-prize-data",
    output_dir="data/netflix",)

print("Path to dataset files:", path)