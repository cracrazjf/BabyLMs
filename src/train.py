from typing import Dict, Any
from psychai.config.training_config import TrainingConfig
from psychai.config.io import update_config
from psychai.trainer.lm_trainer import LM_Trainer
import numpy as np

def main():
    cfg = TrainingConfig()
    updates = {
        "model": {
            "name": "lstm_200",
            "path": "./models/lstm-200",
            "tokenizer_path": "./tokenizer/full_bates_tokenizer",
            "customized_model": True,
        },
        "data": {
            "train_path": "./data/childes/text_docs",
            "val_path": None,
            "shuffle_dataset": False,
            "shuffle_dataloader": True,
            "stride": 1,
            "pad_left": True,
            "drop_last": False,
            "batch_size": 10,
            "sequence_length": 25,
            "data_process_batch_size": 20,
            "data_process_num_proc": 4
        },
        "optim": {
            "lr": 0.01,
            "optimizer": "adamw"
        },
        "experiment_name": "childes",
        "experiment_directory": "./childes",
        "training_method": "bptt",
        "num_runs": 1,
        "num_epochs": 1,
        "seed": 66
    }
    cfg = update_config(cfg, updates)
    trainer = LM_Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
    