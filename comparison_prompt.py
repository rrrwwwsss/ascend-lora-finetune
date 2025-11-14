import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import re
import json
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


# ======================
# 加载模型和处理器
# ======================
model_name_ori = "/data02/Qwen2.5-VL-7B-Instruct"  # 原始模型
model_name_finetuned = "./models/output/qwen2_vl_finetuned/checkpoint-176"    # 微调模型

tokenizer = AutoTokenizer.from_pretrained(model_name_ori, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name_ori)

model_ori = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_ori,
                                                             torch_dtype=torch.bfloat16,
                                                             device_map="auto",  # 自动分配到 GPU（更高效）
                                                             trust_remote_code=True)

# 微调后的模型
# val_peft_model = PeftModel.from_pretrained(model_ori, model_id="./models/output/qwen2_vl_finetuned/checkpoint-176",is_trainable=False)
print("Model Loaded")

# ======================
# 推理函数
# ======================
def predict(messages, model):
    print("构造输入信息：")
    print(messages)
    print(">>> 开始生成 text")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(">>> 处理图像信息")
    image_inputs, _ = process_vision_info(messages)

    print(">>> 构造输入数据")
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

    print(">>> 移动到 GPU")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(">>> 开始推理")
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    print(">>> 解码输出")
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("Generated Text:", generated_text)

    print(">>> 提取 JSON 结果")

    return pattern_recognition(generated_text)

if __name__ == "__main__":

    # ======================
    # 加载测试集
    # ======================
    test_csv = "./train_data/test_data_prompt_deal.csv"
    df_test = pd.read_csv(test_csv)
    df_test = df_test.head(10)  # 只取前30条数据

    # 提取图片路径和真实标签
    test_images = ["./train_data/pic_pack/" + p for p in df_test["image_path"].tolist()]
    test_images = [p.replace("\\", "/") for p in test_images]
    print(test_images)
    test_prompts = df_test["prompt_text"].tolist()
    test_prompts_ori = df_test["prompt_ori"].tolist()


    def pattern_recognition(response):

        result_data = {"result": "错误"}  # 默认值

        if isinstance(response, dict):
            print("response 类型: dict")
            result_data = response
        elif isinstance(response, str):
            print("response 类型: str")
            # 提取第一个 {...} JSON
            match = re.findall(r'\{.*?\}', response, re.S)

            print("match", match)
            if match:
                last_match = match[-1]  # 获取最后一个匹配项
                print("last_match", last_match)
                result_match  = re.search(r'"result":\s*"([^"]+)"', last_match)
                try:
                    if result_match:
                        result_data = result_match.group(1)  # 获取捕获的结果（"yes" 或 "no"）
                        print("提取的 result 值:", result_data)
                        result_data = {"result": result_data}
                    else:
                        print("没有找到 'result' 字段")
                        return None
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析错误: {e}, 原始数据: {last_match}")
            else:
                print("未找到任何有效的 JSON 数据")
        elif isinstance(response, list) and response and isinstance(response[0], str):
            print("response 类型: list[str]")
            match = re.search(r'\{.*?\}', response[0], re.S)
            if match:
                try:
                    result_data = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析错误: {e}, 原始数据: {match.group(0)}")
        else:
            print("response 类型未知:", type(response))
        result_data = result_data["result"]
        return result_data

    # 从 CSV 解析标签
    test_labels = [pattern_recognition(x) for x in df_test["model_result"]]
    # ======================
    # 遍历测试集做对比
    # ======================

    all_predictions = []
    assert len(test_images) == len(test_prompts) == len(test_labels), \
        f"Length mismatch: images={len(test_images)}, prompts={len(test_prompts)}, labels={len(test_labels)}"
    # 串行
    for img, prompt,true_label,prompts_ori in zip(test_images, test_prompts,test_labels,test_prompts_ori):


        msg_ori = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},  # 图像输入
                {"type": "text", "text": prompts_ori}  # 文本提示（已清理）
            ]
        }]

        msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},  # 图像输入
                {"type": "text", "text": prompt}  # 文本提示（已清理）
            ]
        }]
        # 打印当前使用的修改后提示文本，用于调试或日志记录
        print('ori_prompt:',prompts_ori)
        print('prompt:',prompt)
        # 使用原始提示词对当前输入进行预测
        result1 = predict(msg_ori, model_ori)
        # 使用调整后的提示词使用相同模型进行预测
        result2 = predict(msg, model_ori)
        print('true',true_label)
        print("ori:", result1)  # 原始模型输出
        print("ft:", result2)  # 微调后模型输出
        element = {
            'true': true_label,
            'ori': result1,
            'ft': result2,
        }
        all_predictions.append(element)
    # 从 all_predictions 中解析出三个列表
    true_labels = [item['true'] for item in all_predictions]
    preds_ori = [item['ori'] for item in all_predictions]
    preds_ft = [item['ft'] for item in all_predictions]
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='macro'),
            "recall": recall_score(y_true, y_pred, average='macro'),
            "f1": f1_score(y_true, y_pred, average='macro')
        }
    print('正确答案：', test_labels)
    print('原模型',preds_ori)
    print('微调后模型',preds_ft)

    metrics_ori = compute_metrics(test_labels, preds_ori)
    metrics_ft = compute_metrics(test_labels, preds_ft)

    print("\n===== 原模型性能 =====")
    print(metrics_ori)
    print("\n===== 微调后模型性能 =====")
    print(metrics_ft)

    with open("model_performance.txt", "w", encoding="utf-8") as f:
        f.write("===== 原模型性能 =====\n")
        f.write(str(metrics_ori) + "\n\n")
        f.write("===== 微调后模型性能 =====\n")
        f.write(str(metrics_ft) + "\n")

    print("结果已保存到 model_performance.txt")