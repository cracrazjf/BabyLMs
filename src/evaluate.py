from psychai.config import EvaluationConfig, update_config
import os
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding
from eval_fn import cloze_eval_fn

def evaluate_lm():
    from psychai.language.lm import TrainingManager
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
        "exp_dir": "./results/gpt2",
        "device": "cpu"
    }
    cfg = update_config(cfg, updates)
    
    tm = TrainingManager(cfg)
    tm.mm.load_model(model_name=cfg.model.name,
                     model_path=cfg.model.path,
                     model_type=cfg.model.model_type,
                     tokenizer_path=cfg.model.tokenizer_path,
                     device=cfg.device,
                     trust_remote_code=cfg.model.trust_remote_code)
    device = next(tm.mm.model.parameters()).device
    print(f"Model loaded on device: {device}")

    dataset = load_dataset(path=cfg.data.test_path, split="train")
    old_cols = dataset.column_names
    pad_token_id = tm.mm.tokenizer.pad_token_id
    if pad_token_id is None:
        tm.mm.tokenizer.pad_token_id = tm.mm.tokenizer.eos_token_id

    def _tokenize_function(batch):
        input_enc = tm.mm.tokenizer(batch["input_text"], add_special_tokens=True, truncation=False)
        return {"input_ids": input_enc["input_ids"]}

    tokenized_dataset = dataset.map(_tokenize_function, 
                                    batched=True, 
                                    batch_size=cfg.data.data_process_batch_size, 
                                    num_proc=cfg.data.data_process_num_proc)

    tokenized_dataset = tokenized_dataset.remove_columns(old_cols)
    tokenized_dataset = tokenized_dataset.add_column("idx", list(range(len(dataset))))
    return cfg, tm, tokenized_dataset

def evaluate_llm():
    from psychai.language.llm import TrainingManager
    cfg = EvaluationConfig()
    updates = {
        "model": {
            # unsloth/Qwen3-8B-Base, unsloth/Meta-Llama-3.1-8B, unsloth/Qwen3-8B-Instruct, unsloth/Meta-Llama-3.1-8B-Instruct
            "name": "unsloth/Qwen3-8Bt",
            "path": f"unsloth/Qwen3-8B",
            "model_type": "qwen3-8b-instruct",
        },
        "data": {
            "test_path": f"/root/autodl-tmp/data/ACL/chat_eval_data",
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
        "exp_name": "qwen3_8b_instruct_evaluation",
        "exp_dir": f"/root/autodl-tmp/results/qwen3_8b_instruct",
        "task": "control",
        "device": "cuda"
    }
    cfg = update_config(cfg, updates)
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

    dataset = load_dataset(path=cfg.data.test_path, split="train")
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
            input_enc = tm.mm.tokenizer(batch["input_text"], add_special_tokens=False, truncation=False)

            return {
                "input_ids": input_enc["input_ids"],}

        tokenized_dataset = dataset.map(_tokenize_function, 
                                        batched=True, 
                                        batch_size=cfg.data.data_process_batch_size, 
                                        num_proc=cfg.data.data_process_num_proc)
    
    tokenized_dataset = tokenized_dataset.remove_columns(old_cols)
    tokenized_dataset = tokenized_dataset.add_column("idx", list(range(len(dataset))))

    return cfg, tm, tokenized_dataset

def main():
    cfg, tm, tokenized_dataset = evaluate_lm()
    os.makedirs(cfg.exp_dir, exist_ok=True)

    collate_fn = DataCollatorWithPadding(tokenizer=tm.mm.tokenizer)
    
    loader = DataLoader(tokenized_dataset, 
                        batch_size=cfg.data.batch_size, 
                        shuffle=False, 
                        num_workers=0,
                        collate_fn=collate_fn)

    tm.evaluate(loader, eval_fn=cloze_eval_fn, epoch=0, eval_path=os.path.join(cfg.exp_dir, "eval_results.jsonl"))
    
if __name__ == "__main__":
    main()