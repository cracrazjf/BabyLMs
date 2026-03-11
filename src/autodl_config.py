import os
def configure_autodl_cache_dirs():
    env_vars = {
        "HF_HOME": "/root/autodl-tmp/cache",
        "HF_HUB_CACHE": "/root/autodl-tmp/cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/autodl-tmp/cache/huggingface/datasets",
        "TRANSFORMERS_CACHE": "/root/autodl-tmp/cache/huggingface/transformers",
        "TORCH_HOME": "/root/autodl-tmp/cache/pytorch_cache",
        "TORCH_HUB": "/root/autodl-tmp/cache/torch_hub",
    }

    for key, path in env_vars.items():
        os.environ[key] = path
        os.makedirs(path, exist_ok=True)

    return env_vars

def configure_hf_endpoint():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

