
from collections import Counter
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed,
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix
import torch
import datetime
import os
import argparse
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve


parser = argparse.ArgumentParser(description="Targeted antimicrobial.")

# Add argument for antimicrobial labels
parser.add_argument('--labels', nargs='+', help='List of labels', default = ['resistance_nitrofurantoin'])
args = parser.parse_args()
labels_list = args.labels



# Set seed for reproducibility
set_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()

# Columns to use from the dataset
use_cols = ['age', 'race', 'veteran', 'gender', 'BMI', 'previous_antibiotic_exposure_cephalosporin',
            'previous_antibiotic_exposure_carbapenem', 'previous_antibiotic_exposure_fluoroquinolone',
            'previous_antibiotic_exposure_polymyxin', 'previous_antibiotic_exposure_aminoglycoside',
            'previous_antibiotic_exposure_nitrofurantoin', 'previous_antibiotic_resistance_ciprofloxacin',
            'previous_antibiotic_resistance_levofloxacin', 'previous_antibiotic_resistance_nitrofurantoin',
            'previous_antibiotic_resistance_sulfamethoxazole', 'resistance_nitrofurantoin',
            'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin', 
            'dept_ER', 'dept_ICU', 'dept_IP', 'dept_OP', 'dept_nan', 'Enterococcus_faecium', 'Staphylococcus_aureus',
            'Klebsiella_pneumoniae', 'Acinetobacter_baumannii', 'Pseudomonas_aeruginosa', 'Enterobacter',
            'organism_other', 'organism_NA', 'additional_note']

data = pd.read_csv("/ibex/project/c2205/AMR_dataset_peijun/integrate/final_all_additional_note_feb14.csv", usecols=use_cols, header=0)
# how many data to load, you can edit the frac
data = data.sample(frac=1, random_state=42)

# Separate features and labels
features = data.drop(columns=["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"])
labels = data[["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"]]

# Convert CSV rows into NLP model's input format
data["input"] = features.apply(lambda row: "; ".join([f"{col.replace('_', ' ')}:{val}" for col, val in row.items()]), axis=1)


# Function to process the data into tokens
def process_func(example, antibiotics):
    MAX_LENGTH = 318
    input_ids, attention_mask, labels = [], [], []
    

    feature_str = example["input"]
    
    # Construct the instruction for the model
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in predicting antibiotic resistance for {antibiotics} based on patient electronic health records. Please output the prediction results.<|im_end|>\n<|im_start|>user\n{feature_str}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        truncation=True, 
        max_length=MAX_LENGTH,  
        padding="max_length", 
    )
    
    # label for prediction
    label = int(example["output"])  # 标签是 0 或 1
    

    input_ids = instruction["input_ids"]
    attention_mask = instruction["attention_mask"]
    labels = label 
    
    # # truncation
    # if len(input_ids) > MAX_LENGTH:
    #     input_ids = input_ids[:MAX_LENGTH]
    #     attention_mask = attention_mask[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def optimize_thresholds(y_true, y_proba):
    thresholds = np.linspace(0, 1, 50)
    best_threshold = 0.0
    best_f1_score = 0.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold


    print(f"Best threshold: {best_threshold}")
    
    return best_threshold

def compute_metrics(eval_preds):
    logits, labels = eval_preds

    y_proba = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)  # 计算类别概率
    best_threshold = optimize_thresholds(labels, y_proba[:,1])

    preds =  (y_proba[:,1] >= best_threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    # compute confusion matrix
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # more measurement
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score_ = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    
    # computiing AUROC and AUPRC
    auroc = roc_auc_score(labels, y_proba[:, 1])  # 假设二分类问题，取正类的概率
    auprc = average_precision_score(labels, y_proba[:, 1])
    

    print()
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("sensitivity:", sensitivity)
    print("specificity:", specificity)
    print("ppv:", ppv)
    print("f1_score:", f1_score_)
    print("AUROC:", auroc)
    print("AUPRC:", auprc)
    

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "f1_score": f1_score_,
    }

# Download Qwen model
snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")


# train 
for label in labels_list:

    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    X = data['input'].tolist()
    Y = labels[label].tolist()
    valid_indices = [i for i, y in enumerate(Y) if not np.isnan(y)]
    X = [X[i] for i in valid_indices]
    Y = [Y[i] for i in valid_indices]
    antimicrobial = label.split('_')[-1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)



    
    def build_messages(x_list, y_list):
        messages = []
        for feature, output in zip(x_list, y_list):
            messages.append({
                "instruction": f"You are an expert in prediction of antimicrobial resistance for {antimicrobial}, and you will receive patients’ electronic health record features. Please output the prediction results.",
                "input": feature,
                "output": output,
            })
        return messages

    train_messages = build_messages(x_train, y_train)
    test_messages = build_messages(x_test, y_test)

    # 训练数据集    
    len_train = len(y_train)
    print("Trainning datasets length:" + str(len_train))
    train_df = pd.DataFrame(train_messages)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, fn_kwargs={"antibiotics": antimicrobial}, remove_columns=train_ds.column_names)

    # 测试数据集
    len_test = len(y_test)
    print("Test datasets length:" + str(len_test))
    test_df = pd.DataFrame(test_messages)
    test_ds = Dataset.from_pandas(test_df)
    test_dataset = test_ds.map(process_func, fn_kwargs={"antibiotics": antimicrobial}, remove_columns=train_ds.column_names)



    # loading claasification model 
    Qwen_model = AutoModelForSequenceClassification.from_pretrained(
        "./qwen/Qwen2-1___5B-Instruct/",
        device_map="auto",
        # torch_dtype=torch.float16,
        num_labels=2,  # classifcation model
    )
    
    Qwen_model.config.pad_token_id = tokenizer.pad_token_id


    config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # change from TaskType.CAUSAL_LM to TaskType.SEQ_CLS
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        # "o_proj",
        # "gate_proj",
        "up_proj",
        "down_proj",
        ],
        inference_mode=False,
        r=32,
        lora_alpha=64,
        modules_to_save=["score", "classifier"],  # add params in classification head as trainable params
        # lora_dropout=0.1    

    )


    model = get_peft_model(Qwen_model, config)


    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("I require grad:", name)
        else:
            print(name)

    # training param
    args = TrainingArguments(
        output_dir="./output_my_training_classification_ubl_March_5_final_complete/" + antimicrobial + "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        per_device_train_batch_size=64,
        per_device_eval_batch_size = 64,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        logging_steps=5,
        num_train_epochs=20,
        learning_rate=3e-5, # from 1e-5 to 3e-4
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=500,
        report_to=["tensorboard"],
        seed=42,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        dataloader_num_workers=4,
        deepspeed=None,
        warmup_ratio=0.1,
    )

    # initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # star training
    trainer.train()