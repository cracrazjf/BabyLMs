from psychai.language.llm import ModelManager
from psychai.config import EvaluationConfig, update_config
import os
import json
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from eval_fn import cat_eval_A, cat_eval_B, cohypo_eval_A, cohypo_eval_B
from prepare_dataset import prepare_evaluation_data, create_counterbalance_data

def main():
    cfg = EvaluationConfig()
    root_dir = "./"
    updates = {
        "model": {
            "name": "unsloth/Meta-Llama-3.1-8B",
            "path": f"unsloth/Meta-Llama-3.1-8B",
            "model_type": "llama",
        },
        "data": {
            "test_path": f"{root_dir}data/childes/eval_data/cohypo_eval_B_prompt1_type1.jsonl",
            "batch_size": 8,
            "data_process_batch_size": 16,
            "data_process_num_proc": 0,
        },
        "logging": {
            "return_weights": False,
            "return_embeddings": True,
        },
        "root_dir": root_dir,
        "exp_name": "llama_8b_evaluation",
        "exp_dir": "./evaluation/llama_8b",
        "task": "cohypo_eval_B_prompt1_type1",
        "layer_type": "h_2",
        "embed_type": "hidden",
        "device": "cpu"
    }
    cfg = update_config(cfg, updates)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    mm = ModelManager()
    mm.load_model(
        model_name=cfg.model.name,
        model_path=cfg.model.path,
        model_type=cfg.model.model_type,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=cfg.model.dtype
    )

    # device = next(tm.mm.model.parameters()).device
    # print(f"Model loaded on device: {device}")

    # dataset = None
    # eval_fn = None
    # if Path(f"{cfg.data.test_path}").exists() is False:
    #     prepare_evaluation_data(eval_type=cfg.task)
        
    # dataset = load_dataset("json", data_files=f"{cfg.data.test_path}", split="train")

    # if "cat_eval_A" in cfg.task:
    #     eval_fn = cat_eval_A
    # elif "cat_eval_B" in cfg.task:
    #     eval_fn = cat_eval_B
    # elif "cohypo_eval_A" in cfg.task:
    #     eval_fn = cohypo_eval_A
    # elif "cohypo_eval_B" in cfg.task:
    #     eval_fn = cohypo_eval_B

    # old_cols = dataset.column_names
    # pad_token_id = tm.mm.tokenizer.pad_token_id
    # if pad_token_id is None:
    #     tm.mm.tokenizer.pad_token_id = tm.mm.tokenizer.eos_token_id

    # def _tokenize_function(batch):
    #     input_text = [prompt + input for prompt, input in zip(batch["prompt"], batch["input"])]
    #     input_enc = tm.mm.tokenizer(input_text, add_special_tokens=False, truncation=False)

    #     return {
    #         "input_ids": input_enc["input_ids"],}
    
    # tokenized_dataset = dataset.map(_tokenize_function, 
    #                                 batched=True, 
    #                                 batch_size=cfg.data.data_process_batch_size, 
    #                                 num_proc=cfg.data.data_process_num_proc)
    
    # tokenized_dataset = tokenized_dataset.remove_columns(old_cols)
    
    # collate_fn = DataCollatorWithPadding(tokenizer=tm.mm.tokenizer)
    
    # loader = DataLoader(tokenized_dataset, 
    #                     batch_size=cfg.data.batch_size, 
    #                     shuffle=False, 
    #                     collate_fn=collate_fn)
    
    
if __name__ == "__main__":
    main()