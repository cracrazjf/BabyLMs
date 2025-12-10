from psychai.config import EvaluationConfig, update_config
from psychai.language.lm import TrainingManager
import os
import json
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding
from eval_fn import cat_eval_A, cat_eval_B, cohypo_eval_A, cohypo_eval_B
from prepare_dataset import prepare_evaluation_data, create_counterbalance_data

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
            "test_path": f"./data/childes/plain_eval_data",
            "batch_size": 8,
            "data_process_batch_size": 16,
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
        "exp_dir": "./generation/gpt2",
        "task": "all",
        "device": "cpu"
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
        dataset = None
        eval_fn = None
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

        def _tokenize_function(batch):
            input_text = [prompt + input for prompt, input in zip(batch["prompt"], batch["input"])]
            input_enc = tm.mm.tokenizer(input_text, add_special_tokens=False, truncation=False)

            return {"input_ids": input_enc["input_ids"]}

        # def _tokenize_function(batch):
        #     cleaned_inputs = []

        #     for prompt, inp, remove in zip(batch["prompt"], batch["input"], batch["c_word"]):
        #         prefix = inp.split(remove)[0]
        #         text = prompt + prefix
        #         cleaned_inputs.append(text)

        #     enc = tm.mm.tokenizer(cleaned_inputs, add_special_tokens=False, truncation=False)

        #     return {"input_ids": enc["input_ids"]}
    
        tokenized_dataset = dataset.map(_tokenize_function, 
                                        batched=True, 
                                        batch_size=cfg.data.data_process_batch_size, 
                                        num_proc=cfg.data.data_process_num_proc)
    
        tokenized_dataset = tokenized_dataset.remove_columns(old_cols)
        
        # df = tokenized_dataset.to_pandas()
        # df["input_ids_tuple"] = df["input_ids"].apply(tuple)
        # df = df.drop_duplicates(subset=["input_ids_tuple"])
        # df = df.drop(columns=["input_ids_tuple"])
        # tokenized_dataset = Dataset.from_pandas(df, preserve_index=False)

        collate_fn = DataCollatorWithPadding(tokenizer=tm.mm.tokenizer)
        
        loader = DataLoader(tokenized_dataset, 
                            batch_size=cfg.data.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn)
        
        results = tm.evaluate(loader, eval_fn=eval_fn, epoch=0)

        with open(cfg.exp_dir + f"/{Path(test_file).stem}_results.jsonl", "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # generations = []
        # for batch in loader:
        #     decoded_text = tm.generate(batch["input_ids"], max_new_tokens=10, temperature=0.0, top_k=0)
        #     generations.extend(decoded_text)

        # with open(cfg.exp_dir + f"/{Path(test_file).stem}_generations.jsonl", "w", encoding="utf-8") as f:
        #     for gen in generations:
        #         rec = {
        #             "generation": gen
        #         }
        #         f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    
if __name__ == "__main__":
    main()