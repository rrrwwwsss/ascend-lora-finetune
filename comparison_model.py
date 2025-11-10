import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import re
import json
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# ======================
# 加载测试集
# ======================
test_csv = "../../train_data/train_data1.csv"
df_test = pd.read_csv(test_csv)

# 提取图片路径和真实标签
test_images = ["../../train_data/pic_pack/" + p for p in df_test["image_path"].tolist()]
test_prompts = df_test["prompt_text"].tolist()
def extract_result(text):
    # 提取最后一个 {"result": "..."} JSON
    match = re.search(r'\{[^{}]*"result"[^{}]*\}', text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("result", "unknown")
        except json.JSONDecodeError:
            return "unknown"
    else:
        return "unknown"

# 从 CSV 解析标签
test_labels = [extract_result(x) for x in df_test["model_result"]]

# ======================
# 加载模型和处理器
# ======================
model_name_ori = "../../../models/Qwens/Qwen2-VL-2B-Instruct/"  # 原始模型
model_name_finetuned = "./models/output/qwen2_vl_finetuned"    # 微调模型

tokenizer = AutoTokenizer.from_pretrained(model_name_ori, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name_ori)

model_ori = Qwen2VLForConditionalGeneration.from_pretrained(model_name_ori,
                                                             torch_dtype=torch.bfloat16,
                                                             device_map="auto",  # 自动分配到 GPU（更高效）
                                                             trust_remote_code=True)
# model_finetuned = Qwen2VLForConditionalGeneration.from_pretrained(model_name_finetuned,
#                                                                   torch_dtype=torch.bfloat16,
#                                                                   device_map="auto",  # 自动分配到 GPU（更高效）
#                                                                   trust_remote_code=True)
# 微调后的模型
val_peft_model = PeftModel.from_pretrained(model_ori, model_id="./models/output/qwen2_vl_finetuned/checkpoint-176")
print("Model Loaded")
# processor = AutoProcessor.from_pretrained(model_name_ori)
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_ori, device_map="auto",
#                                                         torch_dtype=torch.bfloat16, trust_remote_code=True, )

# def predict(messages, model):
#     # 准备推理
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")
#
#     # 生成输出
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#
#     return output_text[0]
# ====================测试模式===================
# # 配置测试参数
# val_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=True,  # 训练模式
#     r=64,  # Lora 秩
#     lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.05,  # Dropout 比例
#     bias="none",
# )

# ======================
# 推理函数
# ======================
def predict(messages, model):
    print(">>> 开始生成 text")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(">>> 处理图像信息")
    image_inputs, _ = process_vision_info(messages)

    print(">>> 构造输入数据")
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

    print(">>> 移动到 GPU")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(">>> 开始推理")
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    print(">>> 解码输出")
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("Generated Text:", generated_text)

    print(">>> 提取 JSON 结果")
    import re, json
    match = re.search(r'```json(.*?)```', generated_text, re.S)
    if match:
        try:
            result_json = json.loads(match.group(1).strip())
            result = result_json.get("result", "no")
        except:
            result = "no"
    else:
        result = "no"

    print(">>> 最终结果:", result)
    return result


#
# # ======================
# # 计算指标
# # ======================
# def compute_metrics(y_true, y_pred):
#     return {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, pos_label="yes"),
#         "recall": recall_score(y_true, y_pred, pos_label="yes"),
#         "f1": f1_score(y_true, y_pred, pos_label="yes")
#     }
#
# metrics_ori = compute_metrics(test_labels, preds_ori)
# metrics_ft = compute_metrics(test_labels, preds_ft)
#
# print("===== 原模型性能 =====")
# print(metrics_ori)
# print("===== 微调后模型性能 =====")
# print(metrics_ft)

if __name__ == "__main__":
    # wajue_question = """
    # **Role:**
    # You are an intelligent assistant capable of accurately identifying road occupation or excavation activities in images.
    #
    # **Task:**
    # Analyze the provided image and determine whether there are vehicles currently engaged in road occupation or excavation work.
    # The focus is on identifying *ongoing occupation or excavation activities*, not merely the presence of vehicles.
    #
    # **To be recognized as an occupation or excavation activity, the following three conditions must all be met:**
    #
    # 1. The occupation or excavation activity itself is visibly taking place.
    # 2. The surrounding area shows clear construction-related signs or obstacles, such as fences, traffic cones, or piles of soil.
    #    *(Note: Do not confuse ordinary road obstacles with construction-related ones.)*
    # 3. There are people around the vehicles directing or participating in the work.
    #
    # **Exclusion criteria:**
    #
    # 1. Ignore large vehicles that are parked or driving within safe zones and not participating in construction.
    # 2. Ignore buildings, pedestrians, toll booths, and road dividers.
    # 3. **Image quality limitation:**
    #    If the image is too blurry, obscured, or poorly lit to make an accurate judgment, respond with **“no.”**
    # """
    # msg = [{"role": "user", "content": [{"type": "image", "image": "../../train_data/pic_pack/wajue/camera_11000000001311523919_20250626_090209.jpg" },
    #                                     {"type": "text", "text": wajue_question}]}]
    # print(predict(msg, model_finetuned))

    # ======================
    # 遍历测试集做对比
    # ======================
    preds_ori = []
    preds_ft = []

    for img, prompt in zip(test_images, test_prompts):
        msg = [{"role": "user", "content": [{"type": "image", "image": "../../train_data/pic_pack/"+img},
                                            {"type": "text", "text": prompt}]}]

        result1 = predict(msg, model_ori)
        result2 = predict(msg, val_peft_model)
        print("ori:", result1)
        print("ft:", result2)
        preds_ori.append(result2)
        preds_ft.append(result1)

    print(preds_ft)