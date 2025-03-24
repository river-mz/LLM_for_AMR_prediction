# -*- coding: utf-8 -*-
import torch
import re
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
from sklearn.metrics import confusion_matrix
import sys

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default = '/ibex/user/xiex/ide/LLM4AMR/llm_final_codes/LLM_for_AMR_prediction/output_march_21/model/lora_adapter', help="Path to the fine-tuned model directory.")
parser.add_argument('--output_dir', type=str, default = './pred_interpretation_output', help="Path to the test dataset CSV file.")
parser.add_argument('--label', type=str, required=True, help="Target antimicrobial label.")
args = parser.parse_args()
with_label = 1
antibiotic = args.label


output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_dataset(csvPath, antibiotic):
    # load the test dataset
    # csvPath: path to your test dataset
    # with_label: whether your test df contain label, 1 or 0
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

    data = pd.read_csv(csvPath,  usecols = use_cols, header=0) 
    

    # 特征工程
    features = data.drop(columns=["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"])
    labels = data[antibiotic].tolist()

    data["input"] = features.apply(lambda row: "; ".join([f"{col.replace('_',' ')}:{val}" for col, val in row.items()]), axis=1)
    

    return data, labels


# loading the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained('../../qwen/Qwen2-1___5B-Instruct/', use_fast=False, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained('../../qwen/Qwen2-1___5B-Instruct/', device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, args.model_dir)
model.to(device)
model.eval()

DatasetPath = f'/ibex/user/xiex/ide/LLM4AMR/training_test_index_in_proj1/dataframe_with_note_for_llm/test_df_{antibiotic}.csv'
data, y_true = load_dataset(DatasetPath, antibiotic)
X_test = data["input"].tolist()

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
    print(response)
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




# 测试阶段
predictions = []
reasons = []
y_label = []
    
test_messages = []
for x in X_test:
    test_messages.append([
        {"role": "system", "content": f"You are an antimicrobial resistance expert for {antibiotic.split('_')[0]} resistance prediction. Analyze the following case and provide:\n1. Prediction: (0 for susceptible, 1 for resistant)\n2. Reasoning: (Key factors influencing the prediction)"},
        {"role": "user", "content": f"Patient features: {x}"}
    ])



# 进行预测
for i, msg in enumerate(test_messages):
    pred, reason = predict_with_reason(msg, model, tokenizer)
    predictions.append(pred)
    reasons.append(reason)
    y_label.append(int(y_true[i]))
    print(f"\nCase {i+1}/{len(test_messages)}")
    print(f"True: {y_true[i]} | Pred: {pred}")
    print(f"Reasoning: {reason[:200]}...")  # 显示前200个字符

# 保存详细结果
result_df = pd.DataFrame({
    "features": X_test,
    "true_label": y_label,
    "prediction": predictions,
    "reasoning": reasons
})
# saving the prediction and reasoning to csv file
result_df.to_csv(os.path.join(output_dir,f"{antibiotic}_predictions.csv"), index=False)

# 计算指标（过滤无效预测）
valid_preds = [(t, p) for t, p in zip(y_label, predictions) if p in (0, 1)]
if len(valid_preds) == 0:
    print("No valid predictions for evaluation")
    sys.exit()

y_true_valid, y_pred_valid = zip(*valid_preds)
cm = confusion_matrix(y_true_valid, y_pred_valid)
tn, fp, fn, tp = cm.ravel()


metrics = {
    "accuracy": (tp + tn) / (tp + tn + fp + fn),
    "recall": tp / (tp + fn),
    "specificity": tn / (tn + fp),
    "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
    "f1": 2*tp / (2*tp + fp + fn)
}

print(f"\nEvaluation Metrics for {antibiotic}:")
for k, v in metrics.items():
    print(f"{k:12}: {v:.4f}")

# 保存指标
with open(f"{antibiotic}_metrics.txt", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll tasks completed!")


