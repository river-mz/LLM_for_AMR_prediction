# -*- coding: utf-8 -*-
import json
import pandas as pd
import torch
import re
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed 
)
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import datetime
import csv
import argparse

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--labels', nargs='+', help='Targeted antimicrobial.')
args = parser.parse_args()
labels_list = args.labels

# 设置随机种子
set_seed(42)
torch.cuda.manual_seed_all(42)

# 数据准备
use_cols = ['age', 'race', 'veteran', 'gender', 'BMI', 'previous_antibiotic_exposure_cephalosporin',
       'previous_antibiotic_exposure_carbapenem', 'previous_antibiotic_exposure_fluoroquinolone',
       'previous_antibiotic_exposure_polymyxin', 'previous_antibiotic_exposure_aminoglycoside',
       'previous_antibiotic_exposure_nitrofurantoin', 'previous_antibiotic_resistance_ciprofloxacin',
       'previous_antibiotic_resistance_levofloxacin', 'previous_antibiotic_resistance_nitrofurantoin',
       'previous_antibiotic_resistance_sulfamethoxazole','resistance_nitrofurantoin', 'resistance_sulfamethoxazole',
       'resistance_ciprofloxacin', 'resistance_levofloxacin', 'source', 'dept_ER', 'dept_ICU',
       'dept_IP', 'dept_OP', 'dept_nan', 'Enterococcus_faecium', 'Staphylococcus_aureus',
       'Klebsiella_pneumoniae', 'Acinetobacter_baumannii', 'Pseudomonas_aeruginosa', 
       'Enterobacter', 'organism_other', 'organism_NA', 'additional_note']

data = pd.read_csv("/path/to/your/dataset.csv", usecols=use_cols, header=0) 
data = data.sample(frac=1/50, random_state=42)

# 特征工程
features = data.drop(columns=labels_list)
labels = data[labels_list]
data["input"] = features.apply(lambda row: "; ".join([f"{col.replace('_',' ')}:{val}" for col, val in row.items()]), axis=1)

# 下载模型
snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left"

# 数据处理函数
def process_func(example, antibiotics):
    MAX_LENGTH = 318
    feature_str = example["input"]
    
    instruction = tokenizer(
        f"<|im_start|>system\nPredict antibiotic resistance for {antibiotics}. Answer with 0 or 1.<|im_end|>\n<|im_start|>user\n{feature_str}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 预测函数（含理由解析）
def predict_with_reason(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", max_length=318, truncation=True).to(device)
    
    # 生成更长响应
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_p=0.9
    )
    
    # 解析响应
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 使用正则表达式提取预测和理由
    prediction, reason = None, ""
    pred_match = re.search(r'Prediction:\s*(\d+)', response, re.IGNORECASE)
    if pred_match:
        prediction = int(pred_match.group(1))
        reason_part = response[pred_match.end():].strip()
        reason_match = re.search(r'Reasoning:\s*(.*)', reason_part, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else reason_part
    else:
        num_match = re.search(r'\b[01]\b', response)
        if num_match:
            prediction = int(num_match.group())
            reason = response.replace(num_match.group(), '', 1).strip()
    
    return prediction, reason

# 主流程
for label in labels_list:
    antimicrobial = label.split('_')[-1]
    print(f"\n=== Processing {antimicrobial} ===")
    
    # 数据准备
    X = data['input'].tolist()
    Y = labels[label].tolist()
    valid_indices = [i for i, y in enumerate(Y) if not np.isnan(y)]
    X = [X[i] for i in valid_indices]
    Y = [Y[i] for i in valid_indices]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 训练数据处理
    def build_messages(x_list, y_list):
        return [{"input": x, "output": y} for x, y in zip(x_list, y_list)]
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(build_messages(x_train, y_train)))
    train_dataset = train_dataset.map(process_func, fn_kwargs={"antibiotics": antimicrobial}, remove_columns=train_dataset.column_names)

    # 模型准备
    model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    
    # LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=16,
    )
    model = get_peft_model(model, peft_config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=f"./output/{antimicrobial}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=15,
        logging_steps=20,
        fp16=True,
        optim="adamw_torch",
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=200,
        seed=42
    )

    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()
    
    # 测试阶段
    predictions = []
    reasons = []
    y_true = []
    
    # 构建测试提示
    test_messages = []
    for x in x_test:
        test_messages.append([
            {"role": "system", "content": f"You are an antimicrobial resistance expert. Analyze the following case and provide:\n1. Prediction (0 for susceptible, 1 for resistant)\n2. Key factors influencing the prediction"},
            {"role": "user", "content": f"Patient features: {x}"}
        ])
    
    # 进行预测
    for i, msg in enumerate(test_messages):
        pred, reason = predict_with_reason(msg, model, tokenizer)
        predictions.append(pred)
        reasons.append(reason)
        y_true.append(int(y_test[i]))
        print(f"\nCase {i+1}/{len(test_messages)}")
        print(f"True: {y_test[i]} | Pred: {pred}")
        print(f"Reasoning: {reason[:200]}...")  # 显示前200个字符

    # 保存详细结果
    result_df = pd.DataFrame({
        "features": x_test,
        "true_label": y_true,
        "prediction": predictions,
        "reasoning": reasons
    })
    result_df.to_csv(f"{antimicrobial}_predictions.csv", index=False)
    
    # 计算指标（过滤无效预测）
    valid_preds = [(t, p) for t, p in zip(y_true, predictions) if p in (0, 1)]
    if len(valid_preds) == 0:
        print("No valid predictions for evaluation")
        continue
    
    y_true_valid, y_pred_valid = zip(*valid_preds)
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "f1": 2*tp / (2*tp + fp + fn)
    }
    
    print(f"\nEvaluation Metrics for {antimicrobial}:")
    for k, v in metrics.items():
        print(f"{k:12}: {v:.4f}")

    # 保存指标
    with open(f"{antimicrobial}_metrics.txt", "w") as f:
        json.dump(metrics, f, indent=2)

print("\nAll tasks completed!")