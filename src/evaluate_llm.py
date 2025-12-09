import os
from utils import configure_autodl_cache_dirs, configure_hf_endpoint
configure_autodl_cache_dirs()
configure_hf_endpoint()
import json
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset
from psychai.language.llm import ModelManager, TrainingManager
from psychai.config import EvaluationConfig, update_config
from transformers import DataCollatorWithPadding
from eval_fn import cat_eval_A, cat_eval_B, cohypo_eval_A, cohypo_eval_B
from prepare_dataset import prepare_evaluation_data, create_counterbalance_data

def main():
    cfg = EvaluationConfig()
    updates = {
        "model": {
            "name": "unsloth/Meta-Llama-3.1-8B",
            "path": f"unsloth/Meta-Llama-3.1-8B",
            "model_type": "llama",
        },
        "data": {
            "test_path": f"/root/autodl-tmp/eval_data",
            "batch_size": 8,
            "data_process_batch_size": 16,
            "data_process_num_proc": 0,
        },
        "logging": {
            "return_weights": False,
            "return_embeddings": True,
            "layer_of_interest": 2,
        },
        "root_dir": "/root/autodl-tmp/",
        "exp_name": "llama_8b_evaluation",
        "exp_dir": f"/root/autodl-tmp/evaluation/llama_8b",
        "task": "all",
        "device": "cuda"
    }
    cfg = update_config(cfg, updates)
    os.makedirs(cfg.exp_dir, exist_ok=True)

    if Path(cfg.data.test_path).is_dir():
        files = [f for f in Path(cfg.data.test_path).iterdir() if f.is_file()]
        cat_eval_A_files = []
        cat_eval_B_files = []
        cohypo_eval_A_files = []
        cohypo_eval_B_files = []
        for f in files:
            if "cat_eval_A" in f.name:
                cat_eval_A_files.append(str(f))
            elif "cat_eval_B" in f.name:
                cat_eval_B_files.append(str(f))
            elif "cohypo_eval_A" in f.name:
                cohypo_eval_A_files.append(str(f))
            elif "cohypo_eval_B" in f.name:
                cohypo_eval_B_files.append(str(f))

    if "cat_eval_A" in cfg.task:
        test_files = cat_eval_A_files
    elif "cat_eval_B" in cfg.task:
        test_files = cat_eval_B_files
    elif "cohypo_eval_A" in cfg.task:
        test_files = cohypo_eval_A_files
    elif "cohypo_eval_B" in cfg.task:
        test_files = cohypo_eval_B_files
    else:
        test_files = cat_eval_A_files + cat_eval_B_files + cohypo_eval_A_files + cohypo_eval_B_files


    tm = TrainingManager(cfg)
    tm.mm.load_model(
        model_name=cfg.model.name,
        model_path=cfg.model.path,
        model_type=cfg.model.model_type,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype
    )

    device = next(tm.mm.model.parameters()).device
    print(f"Model loaded on device: {device}")

    for test_file in test_files:
        print(f"Evaluating on file: {test_file}")
        cfg.data.test_path = test_file
        dataset = None
        eval_fn = None
        if Path(f"{cfg.data.test_path}").exists() is False:
            prepare_evaluation_data(eval_type=cfg.task)
            
        dataset = load_dataset("json", data_files=f"{test_file}", split="train")

        if "cat_eval_A" in Path(test_file).name:
            eval_fn = cat_eval_A
        elif "cat_eval_B" in Path(test_file).name:
            eval_fn = cat_eval_B
        elif "cohypo_eval_A" in Path(test_file).name:
            eval_fn = cohypo_eval_A
        elif "cohypo_eval_B" in Path(test_file).name:
            eval_fn = cohypo_eval_B

        old_cols = dataset.column_names
        pad_token_id = tm.mm.tokenizer.pad_token_id
        if pad_token_id is None:
            tm.mm.tokenizer.pad_token_id = tm.mm.tokenizer.eos_token_id
        tm.mm.tokenizer.padding_side = "right"

        def _tokenize_function(batch):
            input_text = [prompt + input for prompt, input in zip(batch["prompt"], batch["input"])]
            input_enc = tm.mm.tokenizer(input_text, add_special_tokens=False, truncation=False)

            return {
                "input_ids": input_enc["input_ids"],}
    
        tokenized_dataset = dataset.map(_tokenize_function, 
                                        batched=True, 
                                        batch_size=cfg.data.data_process_batch_size, 
                                        num_proc=cfg.data.data_process_num_proc)
        
        tokenized_dataset = tokenized_dataset.remove_columns(old_cols)

        
        collate_fn = DataCollatorWithPadding(tokenizer=tm.mm.tokenizer)
        
        loader = DataLoader(tokenized_dataset, 
                            batch_size=cfg.data.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)
        
        results = tm.evaluate(loader, eval_fn=eval_fn, epoch=0)

        with open(cfg.exp_dir + f"/{Path(test_file).stem}_results.jsonl", "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
if __name__ == "__main__":
    main()