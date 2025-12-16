import re
import os
import random
import json
import copy
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from psychai.language import load_any_as_chat
from psychai.language.tokenizer import make_normalizer, make_pretokenizer, train_tokenizer, wrap_tokenizer, print_tokenizer

def prepare_childes_data():
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

def create_raw_data(path="./data/ACL/LLM_Categories_stim.xlsx"):
    df_dict = pd.read_excel(path, sheet_name=None)
    category_dict = dict(zip(df_dict["probes"]["instance"], df_dict["probes"]["category"]))

    raw_df = copy.deepcopy(df_dict["cohyponyms"])
    cohypo_cols = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
    for i, c in enumerate(cohypo_cols, start=1):
        raw_df[f"category{i}"] = raw_df[c].map(category_dict)

    raw_df.to_json("./data/ACL/stimuli_with_categories.jsonl", orient="records", lines=True)

def prepare_evaluation_data(eval_type: str, 
                            category_file: str = "./data/ACL/stimuli_with_categories.jsonl",
                            raw_file: str = "./data/ACL/LLM_Categories_stim.xlsx",
                            output_dir: str = "./data/ACL/plain_eval_data"):
    records = []
    try:
        with open(category_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except:
        create_raw_data()

    df_dict = pd.read_excel(raw_file, sheet_name=None)
    probe_determiner_dict = dict(zip(df_dict["probes"]["instance"], df_dict["probes"]["probe_determiner"].fillna("")))
    category_determiner_dict = dict(zip(df_dict["categories"]["category"], df_dict["categories"]["category_determiner"].fillna("")))

    os.makedirs(output_dir, exist_ok=True)

    def create_superordinate_A():
        metaprompts = {"metaprompt1": "Please complete the following sentence about the category label for the word that is provided. Respond as concisely as possible. ",
                       "metaprompt2": "Please complete the following sentence naturally. ",
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is {Y} {Z}.",
                "prompt2": "{W} {X} is a kind of {Z}.",
                "prompt3": "{W} {X} is a type of {Z}.",
                "prompt4": "{W} {X} belongs to the category {Z}.",
                "prompt5": "{W} {X} is classified as {Y} {Z}.",
                "negated_prompt1": "{W} {X} is not {Y} {Z}.",
                "negated_prompt2": "{W} {X} is not a kind of {Z}.",
                "negated_prompt3": "{W} {X} is not a type of {Z}.",
                "negated_prompt4": "{W} {X} does not belong to the category {Z}.",
                "negated_prompt5": "{W} {X} is not classified as {Y} {Z}.",
                "control_prompt1": "{X} _ {Z}.",
                "control_prompt2": "{X}: {Z}.",
                "control_prompt3": "{X} -> {Z}.",
                "control_prompt4": "{X} — {Z}.",
                "control_prompt5": "{X} and {Z}."
        }
        
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/superordinate_A_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["category1", "category3", "category4"]
                        for i in range(len(conditions)):
                            probe = rec["probe"]
                            probe_determiner = probe_determiner_dict[probe]
                            category_determiner = category_determiner_dict[rec[conditions[i]]]
                            category_text = rec[conditions[i]]

                            input_text = prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner, Z=category_text).strip()
                            input_text = " ".join(input_text.split())

                            clean_prompt = prompt_template.split("{Z}", 1)[0].format(W=probe_determiner,X=probe, Y="CATEGORY_DETERMINER").strip()

                            cat_space_search = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(category_text)}(?![A-Za-z0-9])"
                            probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                            

                            target = " " + category_text if re.search(cat_space_search, input_text) else category_text
                            probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                            f.write(json.dumps({
                                "relationship": "superordinate",
                                "task": "cloze",
                                "condition": conditions[i],
                                "meta_prompt_key": metaprompt_type,
                                "prompt_key": prompt_type,
                                "combined_prompt": metaprompt + clean_prompt,
                                "input_text": metaprompt + input_text,
                                "probe": probe,
                                "comparison": category_text,
                                "target": target,
                            }, ensure_ascii=False) + "\n")

    if "superordinate_A" in eval_type:
        create_superordinate_A()
        
    def create_superordinate_B():
        metaprompts = {"metaprompt1": """Please answer the following question about the whether the provided word belongs to the stated category. Respond by saying only "True" or "False". """,
                       "metaprompt2": """Please answer the following question. Respond by saying only "True" or "False". """,
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is {Y} {Z}. {A}",
                "prompt2": "{W} {X} is a kind of {Z}. {A}",
                "prompt3": "{W} {X} is a type of {Z}. {A}",
                "prompt4": "{W} {X} belongs to the category {Z}. {A}",
                "prompt5": "{W} {X} is classified as {Y} {Z}. {A}",
                "negated_prompt1": "{W} {X} is not {Y} {Z}. {A}",
                "negated_prompt2": "{W} {X} is not a kind of {Z}. {A}",
                "negated_prompt3": "{W} {X} is not a type of {Z}. {A}",
                "negated_prompt4": "{W} {X} does not belong to the category {Z}. {A}",
                "negated_prompt5": "{W} {X} is not classified as {Y} {Z}. {A}",
                "control_prompt1": "{X} _ {Z}. {A}",
                "control_prompt2": "{X}: {Z}. {A}",
                "control_prompt3": "{X} -> {Z}. {A}",
                "control_prompt4": "{X} — {Z}. {A}",
                "control_prompt5": "{X} and {Z}. {A}"
        }
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/superordinate_B_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["category1", "category3", "category4"]
                        for i in range(2):
                            for j in range(len(conditions)):
                                probe = rec["probe"]
                                probe_determiner = probe_determiner_dict[probe]
                                category_determiner = category_determiner_dict[rec[conditions[j]]]
                                category_text = rec[conditions[j]]

                                input_text = prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner, Z=category_text, A="True" if i == 0 else "False").strip()
                                input_text = " ".join(input_text.split())
                                clean_prompt = prompt_template.split("{A}", 1)[0].format(W=probe_determiner, X=probe, Y=category_determiner, Z=category_text).strip()
                                clean_prompt = " ".join(clean_prompt.split())
                                
                                probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                                probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                                target = " True" if i == 0 else " False"
                                f.write(json.dumps({
                                    "relationship": "superordinate",
                                    "task": "verification",
                                    "condition": conditions[j],
                                    "meta_prompt_key": metaprompt_type,
                                    "prompt_key": prompt_type,
                                    "combined_prompt": metaprompt + clean_prompt,
                                    "input_text": metaprompt + input_text,
                                    "probe": probe,
                                    "comparison": category_text,
                                    "target": target
                                }, ensure_ascii=False) + "\n")

    if "superordinate_B" in eval_type:
        create_superordinate_B()

    def create_cohyponym_A():
        metaprompts = {"metaprompt1": "Please complete the following sentence about words and whether they belong to the same category. Respond as concisely as possible. ",
                       "metaprompt2": "Please complete the following sentence naturally. ",
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is like {Y} {Z}.",
                "prompt2": "{W} {X} is similar to {Y} {Z}.",
                "prompt3": "Two words that belong to the same category are {X} and {Z}.",
                "prompt4": "Another word that belongs to the same category as {X} is {Z}.",
                "prompt5": "{X} is the same type of thing as {Z}.",
                "negated_prompt1": "{W} {X} is not like {Y} {Z}.",
                "negated_prompt2": "{W} {X} is not similar to {Y} {Z}.",
                "negated_prompt3": "Two words that do not belong to the same category are {X} and {Z}.",
                "negated_prompt4": "Another word that does not belong to the same category as {X} is {Z}.",
                "negated_prompt5": "{X} is not the same type of thing as {Z}.",
                "control_prompt1": "{X} _ {Z}.",
                "control_prompt2": "{X}: {Z}.",
                "control_prompt3": "{X} -> {Z}.",
                "control_prompt4": "{X} — {Z}.",
                "control_prompt5": "{X} and {Z}."
        }
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/cohyponym_A_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
                        for i in range(len(conditions)):
                            probe = rec["probe"]
                            probe_determiner = probe_determiner_dict[probe]
                            cohyp_determiner = probe_determiner_dict[rec[conditions[i]]]
                            cohypo_text = rec[conditions[i]]

                            input_text = prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner, Z=cohypo_text).strip()
                            input_text = " ".join(input_text.split())
                            clean_prompt = prompt_template.split("{Z}", 1)[0].format(W=probe_determiner, X=probe, Y="COHYP_DETERMINER").strip()

                            cohypo_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(cohypo_text)}(?![A-Za-z0-9])"
                            probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                            
                            target = " " + cohypo_text if re.search(cohypo_search_pattern, input_text) else cohypo_text
                            probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                            f.write(json.dumps({
                                "relationship": "cohyponym",
                                "task": "cloze",
                                "condition": conditions[i],
                                "meta_prompt_key": metaprompt_type,
                                "prompt_key": prompt_type,
                                "combined_prompt": metaprompt + clean_prompt,
                                "input_text": metaprompt + input_text,
                                "probe": probe,
                                "comparison": cohypo_text,
                                "target": target,
                            }, ensure_ascii=False) + "\n")

    if "cohyponym_A" in eval_type:
        create_cohyponym_A()

    def create_cohyponym_B():
        metaprompts = {"metaprompt1": """Please answer the following question about whether the two words belong to the same category. Respond by saying only "True" or "False". """,
                       "metaprompt2": """Please answer the following question. Respond by saying only "True" or "False". """,
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is like {Y} {Z}. Answer: {A}",
                "prompt2": "{W} {X} is similar to {Y} {Z}. Answer: {A}",
                "prompt3": "Two words that belong to the same category are {X} and {Z}. Answer: {A}",
                "prompt4": "Another word that belongs to the same category as {X} is {Z}. Answer: {A}",
                "prompt5": "{X} is the same type of thing as {Z}. Answer: {A}",
                "negated_prompt1": "{W} {X} is not like {Y} {Z}. Answer: {A} {A}",
                "negated_prompt2": "{W} {X} is not similar to {Y} {Z}. Answer: {A}",
                "negated_prompt3": "Two words that do not belong to the same category are {X} and {Z}. Answer: {A}",
                "negated_prompt4": "Another word that does not belong to the same category as {X} is {Z}. Answer: {A}",
                "negated_prompt5": "{X} is not the same type of thing as {Z}. Answer: {A}",
                "control_prompt1": "{X} _ {Z}. Answer: {A}",
                "control_prompt2": "{X}: {Z}. Answer: {A}",
                "control_prompt3": "{X} -> {Z}. Answer: {A}",
                "control_prompt4": "{X} — {Z}. Answer: {A}",
                "control_prompt5": "{X} and {Z}. Answer: {A}"
        }
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/cohyponym_B_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
                        for i in range(2):
                            for j in range(len(conditions)):
                                probe = rec["probe"]
                                probe_determiner = probe_determiner_dict[probe]
                                cohyp_determiner = probe_determiner_dict[rec[conditions[j]]]
                                cohypo_text = rec[conditions[j]]

                                input_text = prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner, Z=cohypo_text, A="True" if i == 0 else "False").strip()
                                input_text = " ".join(input_text.split())
                                clean_prompt = prompt_template.split("{A}", 1)[0].format(W=probe_determiner, X=probe, Y=cohyp_determiner, Z=cohypo_text).strip()
                                clean_prompt = " ".join(clean_prompt.split())

                                target = " True" if i == 0 else " False"
                                probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                                probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                                f.write(json.dumps({
                                    "relationship": "cohyponym",
                                    "task": "verification",
                                    "condition": conditions[j],
                                    "meta_prompt_key": metaprompt_type,
                                    "prompt_key": prompt_type,
                                    "combined_prompt": metaprompt + clean_prompt,
                                    "input_text": metaprompt + input_text,
                                    "probe": probe,
                                    "comparison": cohypo_text,
                                    "target": target,
                                }, ensure_ascii=False) + "\n")
                
    if "cohyponym_B" in eval_type:
        create_cohyponym_B()

    if eval_type == "all":
        create_superordinate_A()
        create_superordinate_B()
        create_cohyponym_A()
        create_cohyponym_B()

def prepare_chat_evaluation_data(eval_type: str, 
                                 category_file: str = "./data/ACL/stimuli_with_categories.jsonl",
                                 raw_file: str = "./data/ACL/LLM_Categories_stim.xlsx",
                                 output_dir: str = "./data/ACL/chat_eval_data"):
    records = []
    try:
        with open(category_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except:
        create_raw_data()

    df_dict = pd.read_excel(raw_file, sheet_name=None)
    probe_determiner_dict = dict(zip(df_dict["probes"]["instance"], df_dict["probes"]["probe_determiner"].fillna("")))
    category_determiner_dict = dict(zip(df_dict["categories"]["category"], df_dict["categories"]["category_determiner"].fillna("")))

    os.makedirs(output_dir, exist_ok=True)

    def create_superordinate_A():
        metaprompts = {"metaprompt1": "Please complete the following sentence about the category label for the word that is provided. Respond as concisely as possible. ",
                       "metaprompt2": "Please complete the following sentence naturally. ",
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is {Y}",
                "prompt2": "{W} {X} is a kind of",
                "prompt3": "{W} {X} is a type of",
                "prompt4": "{W} {X} belongs to the category",
                "prompt5": "{W} {X} is classified as {Y}",
                "negated_prompt1": "{W} {X} is not {Y}",
                "negated_prompt2": "{W} {X} is not a kind of",
                "negated_prompt3": "{W} {X} is not a type of",
                "negated_prompt4": "{W} {X} does not belong to the category",
                "negated_prompt5": "{W} {X} is not classified as {Y}",
                "control_prompt1": "{X} _ ",
                "control_prompt2": "{X}: ",
                "control_prompt3": "{X} -> ",
                "control_prompt4": "{X} — ",
                "control_prompt5": "{X} and "
        }
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/superordinate_A_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["category1", "category3", "category4"]
                        for i in range(len(conditions)):
                            probe = rec["probe"]
                            probe_determiner = probe_determiner_dict[probe]
                            category_determiner = category_determiner_dict[rec[conditions[i]]]
                            category_text = rec[conditions[i]]

                            input_text = prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner).strip()
                            input_text = " ".join(input_text.split())
                            clean_prompt = prompt_template.split("{Z}", 1)[0].format(W=probe_determiner,X=probe, Y="CATEGORY_DETERMINER").strip()

                            target = category_text
                            probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                            probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                            f.write(json.dumps({
                                "relationship": "superordinate",
                                "task": "cloze",
                                "condition": conditions[i],
                                "meta_prompt_key": metaprompt_type,
                                "prompt_key": prompt_type,
                                "combined_prompt": metaprompt + clean_prompt,
                                "input_text": [
                                        {"role": "user", "content": metaprompt + input_text},
                                        {"role": "assistant", "content": f"{target}."}
                                        ],
                                "probe": probe,
                                "comparison": category_text,
                                "target": target,
                            }, ensure_ascii=False) + "\n")

    if "superordinate_A" in eval_type:
        create_superordinate_A()

    def create_superordinate_B():
        metaprompts = {"metaprompt1": """Please answer the following question about the whether the provided word belongs to the stated category. Respond by saying only "True" or "False". """,
                       "metaprompt2": """Please answer the following question. Respond by saying only "True" or "False". """,
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is {Y} {Z}.",
                "prompt2": "{W} {X} is a kind of {Z}.",
                "prompt3": "{W} {X} is a type of {Z}.",
                "prompt4": "{W} {X} belongs to the category {Z}.",
                "prompt5": "{W} {X} is classified as {Y} {Z}.",
                "negated_prompt1": "{W} {X} is not {Y} {Z}.",
                "negated_prompt2": "{W} {X} is not a kind of {Z}.",
                "negated_prompt3": "{W} {X} is not a type of {Z}.",
                "negated_prompt4": "{W} {X} does not belong to the category {Z}.",
                "negated_prompt5": "{W} {X} is not classified as {Y} {Z}.",
                "control_prompt1": "{X}_{Z}.",
                "control_prompt2": "{X}: {Z}.",
                "control_prompt3": "{X} -> {Z}.",
                "control_prompt4": "{X} — {Z}.",
                "control_prompt5": "{X} and {Z}."
        }
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/superordinate_B_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["category1", "category3", "category4"]
                        for i in range(2):
                            for j in range(len(conditions)):
                                probe = rec["probe"]
                                probe_determiner = probe_determiner_dict[probe]
                                category_determiner = category_determiner_dict[rec[conditions[j]]]
                                category_text = rec[conditions[j]]

                                input_text = prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner, Z=category_text).strip()
                                input_text = " ".join(input_text.split())
                                clean_prompt = prompt_template.format(W=probe_determiner, X=probe, Y=category_determiner, Z=category_text).strip()
                                clean_prompt = " ".join(clean_prompt.split())

                                target = "True" if i == 0 else "False"
                                probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                                probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                                f.write(json.dumps({
                                    "relationship": "superordinate",
                                    "task": "verification",
                                    "condition": conditions[j],
                                    "meta_prompt_key": metaprompt_type,
                                    "prompt_key": prompt_type,
                                    "combined_prompt": metaprompt + clean_prompt,
                                    "input_text": [
                                        {"role": "user", "content": metaprompt + input_text},
                                        {"role": "assistant", "content": target}],
                                    "probe": probe,
                                    "comparison": category_text,
                                    "target": target
                                }, ensure_ascii=False) + "\n")
    if "superordinate_B" in eval_type:
        create_superordinate_B()

    def create_cohyponym_A():
        metaprompts = {"metaprompt1": "Please complete the following sentence about words and whether they belong to the same category. Respond as concisely as possible. ",
                       "metaprompt2": "Please complete the following sentence naturally. ",
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is like {Y}",
                "prompt2": "{W} {X} is similar to {Y}",
                "prompt3": "Two words that belong to the same category are {X} and",
                "prompt4": "Another word that belongs to the same category as {X} is",
                "prompt5": "{X} is the same type of thing as",
                "negated_prompt1": "{W} {X} is not like {Y}",
                "negated_prompt2": "{W} {X} is not similar to {Y}",
                "negated_prompt3": "Two words that do not belong to the same category are {X} and",
                "negated_prompt4": "Another word that does not belong to the same category as {X} is",
                "negated_prompt5": "{X} is not the same type of thing as",
                "control_prompt1": "{X}_",
                "control_prompt2": "{X}: ",
                "control_prompt3": "{X} -> ",
                "control_prompt4": "{X} — ",
                "control_prompt5": "{X} and "
        }
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/cohyponym_A_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
                        for i in range(len(conditions)):
                            probe = rec["probe"]
                            probe_determiner = probe_determiner_dict[probe]
                            cohyp_determiner = probe_determiner_dict[rec[conditions[i]]]
                            cohypo_text = rec[conditions[i]]

                            input_text = prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner).strip()
                            input_text = " ".join(input_text.split())
                            clean_prompt = prompt_template.format(W=probe_determiner, X=probe, Y="COHYP_DETERMINER").strip()

                            target =cohypo_text
                            probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                            probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                            f.write(json.dumps({
                                "relationship": "cohyponym",
                                "task": "cloze",
                                "condition": conditions[i],
                                "meta_prompt_key": metaprompt_type,
                                "prompt_key": prompt_type,
                                "combined_prompt": metaprompt + clean_prompt,
                                "input_text": [
                                        {"role": "user", "content": metaprompt + input_text},
                                        {"role": "assistant", "content": f"{target}."}],
                                "probe": probe,
                                "comparison": cohypo_text,
                                "target": target,
                            }, ensure_ascii=False) + "\n")

    if "cohyponym_A" in eval_type:
        create_cohyponym_A()

    def create_cohyponym_B():
        metaprompts = {"metaprompt1": """Please answer the following question about whether the two words belong to the same category. Respond by saying only "True" or "False". """,
                       "metaprompt2": """Please answer the following question. Respond by saying only "True" or "False". """,
                       "metaprompt3": ""}

        prompt_templates = {
                "prompt1": "{W} {X} is {Y} {Z}.",
                "prompt2": "{W} {X} is a kind of {Z}.",
                "prompt3": "{W} {X} is a type of {Z}.",
                "prompt4": "{W} {X} belongs to the category {Z}.",
                "prompt5": "{W} {X} is classified as {Y} {Z}.",
                "negated_prompt1": "{W} {X} is not {Y} {Z}.",
                "negated_prompt2": "{W} {X} is not a kind of {Z}.",
                "negated_prompt3": "{W} {X} is not a type of {Z}.",
                "negated_prompt4": "{W} {X} does not belong to the category {Z}.",
                "negated_prompt5": "{W} {X} is not classified as {Y} {Z}.",
                "control_prompt1": "{X}_{Z}.",
                "control_prompt2": "{X}: {Z}.",
                "control_prompt3": "{X} -> {Z}.",
                "control_prompt4": "{X} — {Z}.",
                "control_prompt5": "{X} and {Z}."
        }
        
        for metaprompt_type, metaprompt in metaprompts.items():
            for prompt_type, prompt_template in prompt_templates.items():
                path = f"{output_dir}/cohyponym_B_{metaprompt_type}_{prompt_type}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        conditions = ["cohyp1", "cohyp2", "cohyp3", "cohyp4"]
                        for i in range(2):
                            for j in range(len(conditions)):
                                probe = rec["probe"]
                                probe_determiner = probe_determiner_dict[probe]
                                cohyp_determiner = probe_determiner_dict[rec[conditions[j]]]
                                cohypo_text = rec[conditions[j]]

                                input_text = prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner, Z=cohypo_text).strip()
                                input_text = " ".join(input_text.split())
                                clean_prompt = prompt_template.format(W=probe_determiner, X=probe, Y=cohyp_determiner, Z=cohypo_text).strip()
                                clean_prompt = " ".join(clean_prompt.split())

                                target = "True" if i == 0 else "False"
                                probe_search_pattern = rf"(?<=\s)(?<![A-Za-z0-9]){re.escape(probe)}(?![A-Za-z0-9])"
                                probe = " " + probe if re.search(probe_search_pattern, metaprompt + input_text) else probe

                                f.write(json.dumps({
                                    "relationship": "cohyponym",
                                    "task": "verification",
                                    "condition": conditions[j],
                                    "meta_prompt_key": metaprompt_type,
                                    "prompt_key": prompt_type,
                                    "combined_prompt": metaprompt + clean_prompt,
                                    "input_text": [
                                        {"role": "user", "content": metaprompt + input_text},
                                        {"role": "assistant", "content": target}
                                    ],
                                    "probe": probe,
                                    "comparison": cohypo_text,
                                    "target": target,
                                }, ensure_ascii=False) + "\n")
                    
    if "cohyponym_B" in eval_type:
        create_cohyponym_B()

    if eval_type == "all":
        create_superordinate_A()
        create_superordinate_B()
        create_cohyponym_A()
        create_cohyponym_B()