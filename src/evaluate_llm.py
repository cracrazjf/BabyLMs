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
from eval_fn import eval_fn
from prepare_dataset import prepare_evaluation_data, create_raw_data, prepare_chat_evaluation_data

def main():
    cfg = EvaluationConfig()
    updates = {
        "model": {
            "name": "unsloth/Meta-Llama-3.1-8B",
            "path": f"unsloth/Meta-Llama-3.1-8B",
            "model_type": "llama3.1-8b",
        },
        "data": {
            "test_path": f"/root/autodl-tmp/data/ACL/plain_eval_data",
            "batch_size": 32,
            "data_process_batch_size": 16,
            "data_process_num_proc": 0,
        },
        "logging": {
            "return_weights": False,
            "return_embeddings": True,
            "layer_of_interest": 0,
        },
        "root_dir": "/root/autodl-tmp/",
        "exp_name": "meta_llama_3.1_8b_base_evaluation",
        "exp_dir": f"/root/autodl-tmp/evaluation/meta_llama_3.1_8b_base",
        "task": "all",
        "device": "cuda"
    }
    cfg = update_config(cfg, updates)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    # prepare_chat_evaluation_data(eval_type=cfg.task, 
    #                              category_file="/root/autodl-tmp/data/ACL/stimuli_with_categories.jsonl", 
    #                              raw_file="/root/autodl-tmp/data/ACL/LLM_Categories_stim.xlsx",
    #                              output_dir=cfg.data.test_path)
    # prepare_evaluation_data(eval_type=cfg.task,
    #                         category_file="/root/autodl-tmp/data/ACL/stimuli_with_categories.jsonl",
    #                         raw_file="/root/autodl-tmp/data/ACL/LLM_Categories_stim.xlsx",
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
        dataset = load_dataset("json", data_files=f"{test_file}", split="train")

        old_cols = dataset.column_names
        pad_token_id = tm.mm.tokenizer.pad_token_id
        if pad_token_id is None:
            tm.mm.tokenizer.pad_token_id = tm.mm.tokenizer.eos_token_id
        tm.mm.tokenizer.padding_side = "right"

        if "instruct" in cfg.model.model_type:
            tm.mm.choose_chat_template()
            if "llama" in cfg.model.model_type:
                def formatting_prompts_func(examples):
                    convos = examples["input_text"]
                    input_ids = [tm.mm.tokenizer.apply_chat_template(convo, 
                                                                    tokenize = True, 
                                                                    add_generation_prompt = False,) for convo in convos]
                    return { "input_ids" : input_ids, }
            elif "qwen" in cfg.model.model_type:
                def formatting_prompts_func(examples):
                    convos = examples["input_text"]
                    input_ids = [tm.mm.tokenizer.apply_chat_template(convo, 
                                                                    tokenize = True, 
                                                                    add_generation_prompt = False,
                                                                    enable_thinking=False) for convo in convos]
                    return { "input_ids" : input_ids, }

            tokenized_dataset = dataset.map(formatting_prompts_func, 
                                            batched = True, 
                                            batch_size=cfg.data.data_process_batch_size, 
                                            num_proc=cfg.data.data_process_num_proc)
        else:
            def _tokenize_function(batch):
                print(batch["input_text"])
                input_enc = tm.mm.tokenizer(batch["input_text"], add_special_tokens=False, truncation=False)

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