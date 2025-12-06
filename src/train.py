from typing import Dict, Any
from psychai.config import TrainingConfig, update_config
from psychai.language.lm import LM_Trainer
from psychai.nn_builder import save_pretrained
import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    cfg = TrainingConfig()
    updates = {
        "model": {
            "name": "gpt-2",
            "path": "./models/gpt-2",
            "model_type": "custom",
            "tokenizer_path": "./tokenizer/gpt2_tokenizer",
            "num_layers": 12,
        },
        "data": {
            "train_path": "./data/childes/text_docs",
            "val_path": None,
            "shuffle_dataset": True,
            "shuffle_dataloader": False,
            "window_size": 25,
            "stride": 1,
            "batch_size": 10,
            "pad_left": True,
            "drop_last": False,
            "data_process_batch_size": 20,
            "data_process_num_proc": 4,
            "num_workers": 4
        },
        "optim": {
            "lr": 0.01,
            "optimizer": "adamw"
        },
        "logging": {
            "interval_strategy": "step",
            "log_interval": 100,
            "eval_interval": 500,
            "return_logits": False,
            "return_weights": False,
            "return_embeddings": False,
            "save_model": True
        },

        "exp_name": "childes",
        "exp_dir": "./childes",
        "bp_method": "bptt",
        "num_runs": 1,
        "num_epochs": 20,
        "seed": 66,
    }
    cfg = update_config(cfg, updates)
    trainer = LM_Trainer(cfg)

    def transformer_weight_init(model):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    std *= (2 * cfg.model.num_layers) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        model.base_model.apply(_init_weights)
        model.base_model.layers["wte"].emb.weight = model.base_model.layers["lm_head"].proj.weight

        
if __name__ == "__main__":
    main()
    