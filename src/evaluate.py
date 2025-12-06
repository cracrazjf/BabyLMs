from psychai.config import EvaluationConfig, update_config
from psychai.language.lm import TrainingManager
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
    root_path = "/root/autodl-tmp/"
    updates = {
        "model": {
            "name": "gpt-2",
            "wrapper": "causal_lm",
            "path": f"{root_path}models/gpt-2",
            "model_type": "custom",
            "tokenizer_path": f"{root_path}tokenizer/gpt2_tokenizer",
            "num_layers": 12,
        },
        "data": {
            "test_path": f"{root_path}",
            "batch_size": 8,
            "data_process_batch_size": 16,
            "data_process_num_proc": 0,
        },
        "logging": {
            "return_weights": False,
            "return_embeddings": True,
        },
        "exp_name": "gpt2_evaluation",
        "exp_dir": "./evaluation/gpt2",
        "task": "summary",
        "layer_type": "h_2",
        "embed_type": "hidden",
        "device": "cuda"
    }
    cfg = update_config(cfg, updates)

    if cfg.task == "summary":
        summary = {}

        for fname in os.listdir(cfg.exp_dir):
            if not fname.endswith(".jsonl"):
                continue

            task_name = fname.replace("_results.jsonl", "").replace(".jsonl", "")

            correct = 0
            total = 0

            with open(os.path.join(cfg.exp_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if "correct" in rec:
                        total += 1
                        if rec["correct"]:
                            correct += 1
                    elif "embed_correct" in rec:
                        total += 1
                        if rec["embed_correct"]:
                            correct += 1

            acc = correct / total if total > 0 else 0.0
            summary[task_name] = acc
        print("Evaluation Summary:")
        with open(os.path.join(cfg.exp_dir, "summary.txt"), "w", encoding="utf-8") as f:
            for task_name, acc in summary.items():
                f.write(f"{task_name}: {acc:.3f}\n")
                print(f"{task_name}: {acc:.3f}")
        return

    tm = TrainingManager(cfg)
    tm.mm.load_model(model_name=cfg.model.name,
                     model_path=cfg.model.path,
                     model_type=cfg.model.model_type,
                     tokenizer_path=cfg.model.tokenizer_path,
                     device=cfg.device,
                     trust_remote_code=cfg.model.trust_remote_code)
    
    os.makedirs(cfg.exp_dir, exist_ok=True)
    # create_counterbalance_data()
    
    device = next(tm.mm.model.parameters()).device
    print(f"Model loaded on device: {device}")

    dataset = None
    eval_fn = None
    if Path(f"{cfg.data.test_path}/eval_data/{cfg.task}.jsonl").exists() is False:
        prepare_evaluation_data(eval_type="cat_eval_A")
    dataset = load_dataset("json", data_files=f"{cfg.data.test_path}/eval_data/{cfg.task}.jsonl", split="train")

    if cfg.task == "cat_eval_A":
        eval_fn = cat_eval_A
    elif cfg.task == "cat_eval_B":
        eval_fn = cat_eval_B
    elif cfg.task == "cohypo_eval_A":
        eval_fn = cohypo_eval_A
    elif cfg.task == "cohypo_eval_B":
        eval_fn = cohypo_eval_B

    old_cols = dataset.column_names
    pad_token_id = tm.mm.tokenizer.pad_token_id
    if pad_token_id is None:
        tm.mm.tokenizer.pad_token_id = tm.mm.tokenizer.eos_token_id

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
    
    tm.evaluate(loader, eval_fn=eval_fn, epoch=0)
    # prompt = "This question is from a simple knowledge book for children, Answer Yes or No. Question: Is a pigeon a vegetable? Answer:"
    # print(tm.generate(prompt=prompt,
    #             max_length=100,
    #             top_k=0,))

    
if __name__ == "__main__":
    main()