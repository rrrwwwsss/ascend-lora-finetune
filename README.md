# Qwen2.5-VL LoRA 微调与性能对比项目（昇腾 910B）

本项目旨在华为昇腾 910B（Ascend 910B）硬件平台上对 **Qwen2.5-VL** 多模态大模型进行 **LoRA（Low-Rank Adaptation）微调**，并系统性地评估微调前后模型在下游任务上的性能表现。

## 功能说明

- **`train_model.py`**：执行 LoRA 微调的主脚本。
- **`comparison_model.py`**：用于加载微调前后的模型，计算并对比以下关键指标：
  - Accuracy（准确率）
  - Precision（精确率）
  - Recall（召回率）
  - F1 Score（F1 分数）

## 使用方式

### 1. 微调模型

请务必指定使用 **NPU 7**，否则将因设备不一致导致运行时错误：

> ❗ **重要提示**：必须设置环境变量 `ASCEND_RT_VISIBLE_DEVICES=7`，否则会报错：  
> `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, npu:7 and npu:!`

执行命令如下：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python train_model.py
```
### 2. 性能对比
#### 2.1 提示词性能对比
运行以下脚本以后台方式比较原始提示词与调整后的提示词性能（在同一个模型上：ori_model）：
```bash
nohup bash -c 'export ASCEND_RT_VISIBLE_DEVICES=7 && python comparison_prompt.py' > log_prompt.txt 2>&1 &
nohup python comparison_prompt.py > log_prompt.txt 2>&1 &
```
- log.txt ：把标准输出重定向到 log.txt 文件；
- 查看是否运行中：
ps -ef | grep comparison_prompt.py
- 查看实时日志：
tail -f log_prompt.txt
- 停止：
kill -9 <PID>
#### 2.2 模型性能对比
微调完成后，运行以下脚本以后台方式比较原始模型与微调后模型的性能：
```bash
nohup bash -c 'export ASCEND_RT_VISIBLE_DEVICES=7 && python comparison_model.py' > log.txt 2>&1 &
nohup python comparison_model.py > log.txt 2>&1 &
```
- log.txt ：把标准输出重定向到 log.txt 文件；
- 查看是否运行中：
ps -ef | grep comparison_model.py
- 查看实时日志：
tail -f log.txt
- 停止：
kill -9 <PID>