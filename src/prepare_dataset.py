import re
import os
import random
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from psychai.language import load_any_as_chat
from psychai.language.tokenizer import make_normalizer, make_pretokenizer, train_tokenizer, wrap_tokenizer, print_tokenizer

def prepare_training_data():
    text_docs_path = Path("./data/childes/text_docs")
    text_docs_path.mkdir(parents=True, exist_ok=True)
    text_documents = [str(f) for f in text_docs_path.rglob("*.txt")]
      
    jsonl_docs_path = Path("./data/childes/jsonl_docs")
    jsonl_docs_path.mkdir(parents=True, exist_ok=True)
    
    normalizer = make_normalizer(lowercase=True)

    def clean_childes_csv():
        terminator_map = {
            "p": ".",
            "q": "?",
            "e": "!",
            "trail off": "…",
            "trail off question": "… ?",
            "interruption": "<i>",
            "self interruption": "<i>",
            "interruption question": "<i>",
            "self interruption question": "<i>",
            "quotation precedes": "<q>",
            "quotation next line": "<q>",
            "broken for coding": "<b>",
        }

        df = pd.read_csv("./data/childes/all_utterances.csv")
        df = df.dropna(subset=["utterance_text"])
        df = df[df.speaker_role != "Target_Child"]
        term_nan_df = df[df["terminator_type"].isna()]
        
        df["terminator_symbol"] = df["terminator_type"].map(terminator_map).fillna("")
        df["sentence_text"] = df["utterance_text"].astype(str) + " " + df["terminator_symbol"]

        grouped_df = df.groupby(["transcript"], as_index=False).agg(target_child_age=("target_child_age", "first"), text=("sentence_text", " ".join))
        grouped_term_nan_df = term_nan_df.groupby(["transcript"], as_index=False).agg(target_child_age=("target_child_age", "first"), text=("utterance_text", " ".join))    

        def clean_filename(s):
            return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(s))

        global_counter = Counter()
        na_counter = Counter()
        for _, row in grouped_df.iterrows():
            transcript = row["transcript"]
            age = row["target_child_age"]
            text = row["text"]

            normalized_text = normalizer.normalize_str(text)

            global_counter.update(normalized_text.split())

            base_name = clean_filename(transcript)
            json_fname = f"{jsonl_docs_path}/{base_name}.jsonl"

            with open(json_fname, "w", encoding="utf-8") as f:
                json_line = json.dumps({
                    "transcript": transcript,
                    "target_child_age": age,
                    "text": text
                }, ensure_ascii=False)
                f.write(json_line + "\n")

            txt_fname = f"{text_docs_path}/{base_name}.txt"
            with open(txt_fname, "w", encoding="utf-8") as f:
                f.write(text)

        for _, row in grouped_term_nan_df.iterrows():
            text = row["text"]
            normalized_text = normalizer.normalize_str(text)
            na_counter.update(normalized_text.split())

        nan_freq_df = pd.DataFrame(na_counter.items(), columns=["word", "frequency"])
        nan_freq_df = nan_freq_df.sort_values("frequency", ascending=False)
        nan_freq_df.to_csv("./data/childes/token_frequencies_nan_terminator.csv", index=False)

        freq_df = pd.DataFrame(global_counter.items(), columns=["word", "frequency"])
        freq_df = freq_df.sort_values("frequency", ascending=False)
        print("Total unique tokens:", len(freq_df))
        freq_df.to_csv("./data/childes/token_frequencies.csv", index=False)

    # if len(text_documents) == 0:
    clean_childes_csv()
    text_documents = [str(f) for f in text_docs_path.rglob("*.txt")]

    try:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer/childes_tokenizer")
    except:
        # pretokenizer = make_pretokenizer(use_whitespace=True)
        pretokenizer = make_pretokenizer(custom_splits=[(" ", "removed")])
        tokenizer = train_tokenizer(files=text_documents, 
                                    model_type="wordlevel", 
                                    normalizer=normalizer, 
                                    pretokenizer=pretokenizer,)
        
        tokenizer = wrap_tokenizer(tokenizer)
        print_tokenizer(tokenizer)
        tokenizer.save_pretrained("./tokenizer/childes_tokenizer")

    token_df = pd.DataFrame(tokenizer.get_vocab().items(), columns=["word", "id"])
    token_df = token_df.sort_values("word")
    token_df.to_csv("./tokenizer/childes_tokenizer/token_ids.csv", index=False)

def create_counterbalance_data(path="./data/childes/flores_stimuli.xlsx"):
    df = pd.read_excel(path)
    object_category_dict = {}
    for idx, row in df.iterrows():
        target = row["Target"]
        category = row["Category"]
        object_category_dict[target] = category

    df["category1"] = df["C1"].map(object_category_dict)
    df["category2"] = df["C2"].map(object_category_dict)
    df["category3"] = df["C3"].map(object_category_dict)
    df["category4"] = df["C4"].map(object_category_dict)

    jsonl_path = "./data/childes/flores_stimuli_with_categories.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            record = {
                "Target": row["Target"],
                "Category": row["Category"],
                "C1": row["C1"],
                "C2": row["C2"],
                "C3": row["C3"],
                "C4": row["C4"],
                "Category1": row["category1"],
                "Category2": row["category2"],
                "Category3": row["category3"],
                "Category4": row["category4"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved:", jsonl_path)

def prepare_evaluation_data(eval_type: str, 
                            category_file: str = "./data/childes/flores_stimuli_with_categories.jsonl",
                            output_dir: str = "./data/childes/plain_eval_data"):
    records = []
    try:
        with open(category_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except:
        create_counterbalance_data()

    def _a_or_an(noun: str) -> str:
        mass_nones = ['makeup', 'toothpaste', 'lotion', 'drums', 'pliers', 'chalk']
        if noun in mass_nones:
            return ""
        else:
            vowels = "aeiou"
            noun = noun.strip().lower()

            if len(noun) == 0:
                return "a"

            return "an" if noun[0] in vowels else "a"
        
    def phrase(noun):
        article = _a_or_an(noun)
        return f"{article} {noun}".strip()

    os.makedirs(output_dir, exist_ok=True)

    def create_cat_eval_A():
        prompts = {"prompt1": "Think of the category this object naturally belongs to. "}

        input_templates = {
                "input1": "{X} is {Y}.",
                "input2": "{X} is a type of {Y}.",
                "input3": "{X} belongs to {Y}."}

        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cat_eval_A_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        categories = [rec["Category"], rec["Category3"], rec["Category4"]]
                        for i in range(3):
                            input_text = template.format(X=phrase(target), Y=phrase(categories[i]))
                            f.write(json.dumps({
                                "prompt": prompt,
                                "input": input_text,
                                "target": target,
                                "category": " " + categories[i],
                            }, ensure_ascii=False) + "\n")

    if "cat_eval_A" in eval_type:
        create_cat_eval_A()
    
    def create_cat_eval_B():
        prompts = {"prompt1": "Answer the following question with Yes or No. "}

        input_templates = {
                "input1": "Is {X} {Y}? Answer: {Z}",
                "input2": "Is {X} a type of {Y}? Answer: {Z}",
                "input3": "Does {X} belong to {Y}? Answer: {Z}"
        }

        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cat_eval_B_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        categories = [rec["Category"], rec["Category3"], rec["Category4"]]
                        for i in range(3):
                            input_text = template.format(X=phrase(target), Y=phrase(categories[i]), Z="Yes" if i == 0 else "No")
                            f.write(json.dumps({
                                "prompt": prompt,
                                "input": input_text,
                                "target": target,
                                "category": categories[i],
                                "answer": " Yes" if i == 0 else " No"
                            }, ensure_ascii=False) + "\n")
    if "cat_eval_B" in eval_type:
        create_cat_eval_B()

    def create_cohypo_eval_A():
        prompts = {"prompt1": "Think of an object that is semantically similar. "}
        input_templates = {
                "input1": "{X} is like {Y}.",
                "input2": "{X} is similar to {Y}.",
                "input3": "{X} equals {Y}.",
        }
        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cohypo_eval_A_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        cs = [rec["C1"], rec["C2"], rec["C3"], rec["C4"]]
                        for i in range(4):
                            input_text = template.format(X=phrase(target), Y=phrase(cs[i]))
                            f.write(json.dumps({
                                "prompt": prompt,
                                "input": input_text,
                                "target": target,
                                "c_word": " " + cs[i]
                            }, ensure_ascii=False) + "\n")
    if "cohypo_eval_A" in eval_type:
        create_cohypo_eval_A()

    def create_cohypo_eval_B():
        prompts = {"prompt1": "Answer the following question with Yes or No. "}
        input_templates = {
                "input1": "Is {X} like {Y}? Answer: {Z}",
                "input2": "Is {X} similar to {Y}? Answer: {Z}",
                "input3": "Does {X} equal {Y}? Answer: {Z}",
        }
        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cohypo_eval_B_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        cs = [rec["C1"], rec["C2"], rec["C3"], rec["C4"]]
                        for i in range(4):
                            input_text = template.format(X=phrase(target), Y=phrase(cs[i]), Z="Yes" if i < 2 else "No")
                            f.write(json.dumps({
                                "prompt": prompt,
                                "input": input_text,
                                "target": target,
                                "c_word": cs[i],
                                "answer": " Yes" if i < 2 else " No"
                            }, ensure_ascii=False) + "\n")
                    
    if "cohypo_eval_B" in eval_type:
        create_cohypo_eval_B()

    if eval_type == "all":
        create_cat_eval_A()
        create_cat_eval_B()
        create_cohypo_eval_A()
        create_cohypo_eval_B()

def prepare_chat_evaluation_data(eval_type: str,
                                 category_file: str = "./data/childes/flores_stimuli_with_categories.jsonl",
                                 output_dir: str = "./data/childes/chat_eval_data"):
    records = []
    try:
        with open(category_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except:
        create_counterbalance_data()

    def _a_or_an(noun: str) -> str:
        mass_nones = ['makeup', 'toothpaste', 'lotion', 'drums', 'pliers', 'chalk']
        if noun in mass_nones:
            return ""
        else:
            vowels = "aeiou"
            noun = noun.strip().lower()

            if len(noun) == 0:
                return "a"

            return "an" if noun[0] in vowels else "a"
        
    def phrase(noun):
        article = _a_or_an(noun)
        return f"{article} {noun}".strip()
    
    os.makedirs(output_dir, exist_ok=True)

    def create_cat_eval_A():
        prompts = {"prompt1": "Think of the category this object naturally belongs to. "}

        input_templates = {
                "input1": "{X} is",
                "input2": "{X} is a type of",
                "input3": "{X} belongs to"}

        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cat_eval_A_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        categories = [rec["Category"], rec["Category3"], rec["Category4"]]
                        for i in range(3):
                            input_text = template.format(X=phrase(target))
                            f.write(json.dumps({
                               "message": [
                                   {"role": "system", "content": "You are a helpful assistant."},
                                   {"role": "user", "content": prompt + input_text},
                                   {"role": "assistant", "content": phrase(categories[i])}
                                 ],
                                "target": target,
                                "category": " " + categories[i],
                            }, ensure_ascii=False) + "\n")

    if "cat_eval_A" in eval_type:
        create_cat_eval_A()

    def create_cat_eval_B():
        prompts = {"prompt1": "Answer the following question with Yes or No. "}

        input_templates = {
                "input1": "Is {X} {Y}?",
                "input2": "Is {X} a type of {Y}?",
                "input3": "Does {X} belong to {Y}?"
        }

        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cat_eval_B_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        categories = [rec["Category"], rec["Category3"], rec["Category4"]]
                        for i in range(3):
                            input_text = template.format(X=phrase(target), Y=phrase(categories[i]))
                            f.write(json.dumps({
                                 "message": [
                                      {"role": "system", "content": "You are a helpful assistant."},
                                      {"role": "user", "content": prompt + input_text},
                                      {"role": "assistant", "content": "Yes" if i == 0 else "No"}
                                    ],
                                "answer": "Yes" if i == 0 else "No",
                                "target": target,
                                "category": categories[i],
                            }, ensure_ascii=False) + "\n")
    if "cat_eval_B" in eval_type:
        create_cat_eval_B()

    def create_cohypo_eval_A():
        prompts = {"prompt1": "Think of an object that is semantically similar. "}
        input_templates = {
                "input1": "{X} is like",
                "input2": "{X} is similar to",
                "input3": "{X} equals",
        }
        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cohypo_eval_A_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        cs = [rec["C1"], rec["C2"], rec["C3"], rec["C4"]]
                        for i in range(4):
                            input_text = template.format(X=phrase(target))
                            f.write(json.dumps({
                                "message": [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": prompt + input_text},
                                    {"role": "assistant", "content": phrase(cs[i])}
                                    ],
                                "target": target,
                                "c_word": cs[i] if _a_or_an(cs[i]) == "" else " " + cs[i]
                            }, ensure_ascii=False) + "\n")
    if "cohypo_eval_A" in eval_type:
        create_cohypo_eval_A()

    def create_cohypo_eval_B():
        prompts = {"prompt1": "Answer the following question with Yes or No. "}
        input_templates = {
                "input1": "Is {X} like {Y}?",
                "input2": "Is {X} similar to {Y}?",
                "input3": "Does {X} equal {Y}?",
        }
        for prompt_name, prompt in prompts.items():
            for type_name, template in input_templates.items():
                path = f"{output_dir}/cohypo_eval_B_{prompt_name}_{type_name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        target = rec["Target"]
                        cs = [rec["C1"], rec["C2"], rec["C3"], rec["C4"]]
                        for i in range(4):
                            input_text = template.format(X=phrase(target), Y=phrase(cs[i]))
                            f.write(json.dumps({
                                "message": [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": prompt + input_text},
                                    {"role": "assistant", "content": "Yes" if i < 2 else "No"}
                                    ],
                                "answer": "Yes" if i < 2 else "No",
                                "target": target,
                                "c_word": cs[i],
                            }, ensure_ascii=False) + "\n")
                    
    if "cohypo_eval_B" in eval_type:
        create_cohypo_eval_B()

    if eval_type == "all":
        create_cat_eval_A()
        create_cat_eval_B()
        create_cohypo_eval_A()
        create_cohypo_eval_B()