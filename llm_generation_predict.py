
# -*- coding: utf-8 -*-
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed 
)
import tensorboard
from sklearn.model_selection import train_test_split 

import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import datetime
import numpy as np
import pandas as pd
import csv

import argparse

parser = argparse.ArgumentParser()

# Add argument for antimicrobial labels
parser.add_argument('--labels', nargs='+', help='Targeted antimicrobial.')
args = parser.parse_args()
labels_list = args.labels
# labels_list = ["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"]


# Set seed for reproducibility
set_seed(42)  
torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache() 



# Columns to use from the dataset
use_cols = ['age', 'race', 'veteran', 'gender', 'BMI', 'previous_antibiotic_exposure_cephalosporin',
       'previous_antibiotic_exposure_carbapenem',
       'previous_antibiotic_exposure_fluoroquinolone',
       'previous_antibiotic_exposure_polymyxin',
       'previous_antibiotic_exposure_aminoglycoside',
       'previous_antibiotic_exposure_nitrofurantoin',
       'previous_antibiotic_resistance_ciprofloxacin',
       'previous_antibiotic_resistance_levofloxacin',
       'previous_antibiotic_resistance_nitrofurantoin',
       'previous_antibiotic_resistance_sulfamethoxazole','resistance_nitrofurantoin', 'resistance_sulfamethoxazole',
       'resistance_ciprofloxacin', 'resistance_levofloxacin', 'source',
        'dept_ER', 'dept_ICU',
       'dept_IP', 'dept_OP', 'dept_nan',
       'Enterococcus_faecium', 'Staphylococcus_aureus',
       'Klebsiella_pneumoniae', 'Acinetobacter_baumannii',
       'Pseudomonas_aeruginosa', 'Enterobacter', 'organism_other',
       'organism_NA', 'additional_note']


data = pd.read_csv("/ibex/project/c2205/AMR_dataset_peijun/integrate/final_all_additional_note_feb14.csv",  usecols = use_cols, header=0) 
data = data.sample(frac=1/50, random_state=42)  # only use a fraction of dataset for debugging 
print(data.shape)


# Separate features and labels
features = data.drop(columns=["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"])
labels = data[["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"]]


# Convert CSV rows into NLP model's input format
data["input"] = features.apply(lambda row: "; ".join([f"{col.replace('_',' ')}:{val}" for col, val in row.items()]), axis=1)

# Download Qwen model
snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")

# Loading tokenizer (map words in sentences into index)
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left"


# Function to process the data into tokens
def process_func(example, antibiotics):
    # transfer the message into tokens, perform the masking, padding, and max length cutting
    MAX_LENGTH = 318
    input_ids, attention_mask, labels = [], [], []
    
   
    feature_str = example["input"]
    
    # Construct the instruction for the model
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in predicting antibiotic resistance for {antibiotics} based on patient electronic healthe records. Please output the prediction results.<|im_end|>\n<|im_start|>user\n{feature_str}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    
    # Construct response for the model
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # Truncate to max length
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", max_length=318, truncation=True).to(device)
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=3)
    print('input len:')
    print(len(model_inputs.input_ids))
    print(model_inputs.input_ids)
    print('output len:')
    print(len(generated_ids))
    print(generated_ids)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ] 
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


X = []
Y = []



# Iterate over each label to predict
for label in labels_list:

    X = data['input'].tolist()
    Y = labels[label].tolist()

    valid_indices = [i for i, y in enumerate(Y) if not np.isnan(y)]

    X = [X[i] for i in valid_indices]
    Y = [Y[i] for i in valid_indices]

    antimicrobial = label.split('_')[-1]


    print(f"Training and evaluating for label: {label}")
    
    running_times = 0
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    running_times = running_times + 1
    print("Running times:" + str(running_times))


    # Function to build messages for the model
    def build_messages(x_list, y_list):
        messages = []
        for feature, output in zip(x_list, y_list):
            messages.append({
                # "instruction": f"You are an expert in prediction of antimicrobial resistance for {antimicrobial}, and you will receive patients’ electronic health record features. Please output the prediction results.",
                "input": feature,
                "output": output,
            })
        return messages

    train_messages = build_messages(x_train, y_train)
    test_messages = build_messages(x_test, y_test)

    #trainning dataset
    len_train = len(y_train)
    print("Trainning datasets length:" + str(len_train))
    train_df = pd.DataFrame(train_messages)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, fn_kwargs={"antibiotics": antimicrobial}, remove_columns=train_ds.column_names)

    # Load the Qwen model
    Qwen_model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/",device_map="auto",  torch_dtype=torch.bfloat16)
    Qwen_model.enable_input_require_grads()

    # using api in peft (param-effective finetuning)
    # config Lora model 1.5B
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        ],
        inference_mode=False,
        r=8,
        lora_alpha=16,
    )
    
    # Combine Qwen model with LoRA
    model = get_peft_model(Qwen_model, config)


    # Set training arguments
    args = TrainingArguments(
        output_dir="./output_my_training_Feb24_nt_final/"+ antimicrobial + "/"+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        per_device_train_batch_size=16,  # batch size
        # per_device_eval_batch_size=1,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,  
        logging_steps=5,
        num_train_epochs=30,
        learning_rate=1e-5,
        gradient_checkpointing=True,
        report_to=["tensorboard"], # save the training process to tensorboard
        seed=42,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        dataloader_num_workers=2,  

    )

     # Set up the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),

    )
    trainer.train()


    # Test dataset
    # len_test = len(test_index)
    len_test = len(y_test)
    print("Test datasets length:" + str(len_test))
    y_label = []
    y_pred = []
    for i in range(0,len_test,1):
        instruction = f"You are an expert in prediction of antimicrobial resistance for {antimicrobial}, and you will receive patients’ electronic health record features. Please output the prediction results.",
        feature = antimicrobial +": " + x_test[i]
        message = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{feature}"},
        ]
        response = predict(message, model, tokenizer)
        y_pred.append(int(float(response)))
        y_label.append(int(float(y_test[i])))

    # Statistics
    cm = confusion_matrix(y_label,y_pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    pre = (TP)/(TP+FP)
    rec = (TP)/(TP+FN)
    acc = (TP + TN) / (TP + FN + TN + FP)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    f1 = 2 * pre * rec / (pre + rec)

    print()
    print("accuracy:", acc)
    print("precision:", pre)
    print("recall:", rec)
    print("f1:", f1)
    print("sensitivity:", sensitivity)
    print("specificity:", specificity)
    print("ppv:", ppv)


    



