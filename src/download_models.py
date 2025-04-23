from huggingface_hub import snapshot_download
import os

os.makedirs("/app/models", exist_ok=True)
snapshot_download(repo_id="HiDream-ai/HiDream-I1-Full", local_dir="/app/models/HiDream-I1-Full", ignore_patterns=["*.md", "*.txt"])
snapshot_download(repo_id="HiDream-ai/HiDream-I1-Dev", local_dir="/app/models/HiDream-I1-Dev", ignore_patterns=["*.md", "*.txt"])
snapshot_download(repo_id="HiDream-ai/HiDream-I1-Fast", local_dir="/app/models/HiDream-I1-Fast", ignore_patterns=["*.md", "*.txt"])
snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", local_dir="/app/models/Meta-Llama-3.1-8B-Instruct", ignore_patterns=["*.md", "*.txt"])
