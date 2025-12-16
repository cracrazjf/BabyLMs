from psychai.config import EvaluationConfig, update_config
from psychai.language.lm import TrainingManager
import os
import json
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding
from eval_fn import eval_fn
from prepare_dataset import prepare_evaluation_data, create_raw_data

def main():
    cfg = EvaluationConfig()
    updates = {
        "model": {
            "name": "gpt-2",
            "wrapper": "causal_lm",
            "path": f"./models/gpt-2",
            "model_type": "custom",
            "tokenizer_path": f"./tokenizer/gpt2_tokenizer",
            "num_layers": 12,
        },
        "data": {
            "test_path": f"./data/ACL/plain_eval_data",
            "batch_size": 32,
            "data_process_batch_size": 125,
            "data_process_num_proc": 0,
        },
        "logging": {
            "return_weights": False,
            "return_embeddings": True,
            "layer_of_interest": "wte",
            "embed_type": "embeddings",
        },
        "root_dir": "./",
        "exp_name": "gpt2_evaluation",
        "exp_dir": "./evaluation/gpt2",
        "task": "cohyponym_B",
        "device": "cpu"
    }
    cfg = update_config(cfg, updates)
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # create_raw_data()

    # prepare_evaluation_data(eval_type=cfg.task,
    #                         category_file="./data/ACL/stimuli_with_categories.jsonl",
    #                         output_dir=cfg.data.test_path)
    # return

    tasks = ["superordinate_A", "superordinate_B", "cohyponym_A", "cohyponym_B"]
    groups = {k: [] for k in tasks}

    for f in Path(cfg.data.test_path).iterdir():
        if f.is_file():
            for k in tasks:
                if k in f.name:
                    groups[k].append(str(f))
                    break

    test_files = next((groups[k] for k in tasks if k in cfg.task),
                    sum(groups.values(), []))
    
    tm = TrainingManager(cfg)
    tm.mm.load_model(model_name=cfg.model.name,
                     model_path=cfg.model.path,
                     model_type=cfg.model.model_type,
                     tokenizer_path=cfg.model.tokenizer_path,
                     device=cfg.device,
                     trust_remote_code=cfg.model.trust_remote_code)
       
    device = next(tm.mm.model.parameters()).device
    print(f"Model loaded on device: {device}")

    for test_file in test_files:
        print(f"Evaluating on file: {test_file}")
        cfg.data.test_path = test_file
        dataset = load_dataset("json", data_files=f"{test_file}", split="train")

        old_cols = dataset.column_names
        pad_token_id = tm.mm.tokenizer.pad_token_id
        if pad_token_id is None:
            tm.mm.tokenizer.pad_token_id = tm.mm.tokenizer.eos_token_id

        def _tokenize_function(batch):
            input_enc = tm.mm.tokenizer(batch["input_text"], add_special_tokens=False, truncation=False)
            return {"input_ids": input_enc["input_ids"]}
    
        tokenized_dataset = dataset.map(_tokenize_function, 
                                        batched=True, 
                                        batch_size=cfg.data.data_process_batch_size, 
                                        num_proc=cfg.data.data_process_num_proc)
    
        tokenized_dataset = tokenized_dataset.remove_columns(old_cols)

        collate_fn = DataCollatorWithPadding(tokenizer=tm.mm.tokenizer)
        
        loader = DataLoader(tokenized_dataset, 
                            batch_size=cfg.data.batch_size, 
                            shuffle=False, 
                            num_workers=0,
                            collate_fn=collate_fn)
        
        results = tm.evaluate(loader, eval_fn=eval_fn, epoch=0)

        with open(cfg.exp_dir + f"/{Path(test_file).stem}_results.jsonl", "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    
if __name__ == "__main__":
    main()