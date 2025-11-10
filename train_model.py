import os
import torch
device = "npu:0"  # 或者通过其他方式动态确定
# torch.npu.set_device("npu:7")
import pandas as pd

from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
# from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
# import swanlab
import json
import torch_npu

# swanlab.login(api_key="oSr42Kdg1W8ZMcQWMAbbj", save=True)
print(torch.npu.device_count())  # 应该输出 1
print(torch.npu.current_device())  # 应该输出 0


def process_func(example):
    device = 'npu:0'
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    output_content = example["model_result"]
    prompt = example["prompt_text"]
    file_path ="./train_data/pic_pack/" + example["image_path"].replace("\\", "/")  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0].tolist() + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0].tolist() + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    labels = torch.tensor(labels).to(device)
    print("设备：", device)
    print(f"input_ids 设备: {input_ids.device}")
    print(f"attention_mask 设备: {attention_mask.device}")
    print(f"labels 设备: {labels.device}")
    print(f"pixel_values 设备: {inputs['pixel_values'].device}")
    print(f"image_grid_thw 设备: {inputs['image_grid_thw'].device}")
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


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
#     # ✅ 正确方式：遍历字典，把每个 tensor 移到 npu
#     inputs = {k: v.to("npu:7") for k, v in inputs.items()}
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
model_name = "/data02/Qwen2.5-VL-7B-Instruct"
# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name,trust_remote_code=True,
    use_fast=False  # 避免 fast tokenizer 警告
)
# trust_remote_code=True 可以拉取远程代码  .cuda()PyTorch 把整个模型放到单个 GPU
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,
#        torch_dtype=torch.bfloat16, trust_remote_code=True,device_map={"": "npu:7"})
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map={"": device},
    torch_dtype=torch.float16,
    trust_remote_code=True,      # 必须！
    local_files_only=True        # 强制本地加载
)
# ✅ 2. 手动迁移整个模型
# model = model.to("npu:7")
device = list(model.parameters())[0].device
# ✅ 强制把 embedding 移动到 npu:7
model.get_input_embeddings().to(device)
if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
    model.get_output_embeddings().to(device)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

print("模型所在npu")
#print(model.hf_device_map)
print(next(model.parameters()).device)
#for name, param in model.named_parameters():
#    print(name, param.device)
# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
import csv
from datasets import Dataset

csv_path = "./train_data/train_data.csv"

data = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

train_dataset = Dataset.from_list(data)
# 数据预处理
train_dataset = train_dataset.map(process_func)

print("预处理后的数据：",train_dataset)
# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, #定义任务类型。
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #指定哪些层的模块将使用 LoRA 进行微调。
    inference_mode=False,  # 训练模式，即 LoRA 会在训练过程中更新对应层的权重；如果设置为 True，则进入 推理模式，此时不会更新参数，只会使用在训练中已经学习到的参数进行推理。
    r=64,  # Lora 秩
    lora_alpha=16,  # 控制LoRA适应部分与原始模型的权重融合比例。较大的 lora_alpha会增加 LoRA 部分的影响力，较小的 lora_alpha 则会让 LoRA 的影响减弱。
    lora_dropout=0.05,  # Dropout 比例，Dropout 可以防止模型过拟合，特别是在微调时，适当的 dropout 可以提高模型的泛化能力。0.05 表示 5% 的概率将被“丢弃”或屏蔽。
    bias="none",#bias="none" 表示 LoRA 不会对层的偏置项（bias）进行修改。
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)
# 模型加载完之后，立刻添加这一行：
# peft_model = peft_model.to(device)
for name, param in peft_model.named_parameters():
  if "lora" in name.lower():
       print(name, param.device)
# 定义输出目录

for name, param in peft_model.named_parameters():
    if str(param.device) != "npu:7":
        print(f"参数 {name} 不在 npu:7，正在迁移...")
        param.data = param.data.to(device)
output_dir = "./models/output/qwen2_vl_finetuned"

# ✅ 递归创建目录（如果已存在，不报错）
os.makedirs(output_dir, exist_ok=True)
# 配置训练参数
args = TrainingArguments(
    output_dir=output_dir,  #存储最终的微调模型
    per_device_train_batch_size=4, #单个设备（GPU/CPU）上一次迭代的 batch 数量，较大的批次可以更好利用GPU，但会占用更多显存
    gradient_accumulation_steps=4, #梯度积累的步骤数，如果显存不够可以设置多个小的批次累计梯度，这个表示梯度在4个小步骤后才进行一次反向传播更新
    logging_steps=10, #每个10步一次日志记录
    logging_first_step=True,#是否在训练的第一步后进行日志记录。
    num_train_epochs=2,#训练的总轮数，将数据集训练多少遍
    save_steps=100,#每 100 个步骤保存一次模型。
    learning_rate=1e-4,#学习率：控制每次参数更新的步长
    save_on_each_node=True, #在每个节点上保存模型。多用于分布式训练
    gradient_checkpointing=True,#开启梯度检查点，可以减少显存的使用，但计算过程会变慢
    report_to="none",#是否将训练日志报告到外部系统
)

from transformers import DataCollatorForSeq2Seq

class NPUDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, padding=True, model=None):
        super().__init__(tokenizer, padding=padding)
        self.model = model  # 传入模型以获取其主设备

    def __call__(self, features):
        # 先用父类方法进行填充
        batch = super().__call__(features)
        # 将所有张量移动到模型的主设备
        if self.model is not None:
            device = self.model.device
            print(device)
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        return batch

# 使用时
data_collator = NPUDataCollator(tokenizer=tokenizer, padding=True, model=peft_model)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # callbacks=[swanlab_callback],
    data_collator=data_collator,  # 使用自定义的 collator
)
# 获取训练数据加载器
train_dataloader = trainer.get_train_dataloader()

# 取一个 batch 并打印设备信息
for step, batch in enumerate(train_dataloader):
    print("\n🔍 当前 Batch 中各张量所在的设备：")
    for k, v in batch.items():
        if hasattr(v, "device"):
            print(f"  {k}: {v.device} (shape: {v.shape})")
        else:
            print(f"  {k}: {type(v)} (无 device 属性)")
    break  # 只看第一个 batch
print("开始训练...")
# 开启模型训练
trainer.train()

# 保存微调后的模型与分词器
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("微调模型保存成功！")


