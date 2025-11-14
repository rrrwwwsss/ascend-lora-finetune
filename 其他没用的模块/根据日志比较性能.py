import re
import json
import pandas as pd

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

test_csv = "../data/test_data.csv"
df_test = pd.read_csv(test_csv)
# 从 CSV 解析标签
test_labels = [extract_result(x) for x in df_test["model_result"]]
print('正确：',test_labels)

import re

# 假设文件路径为 'log.txt'
file_path = 'log.txt'

# 初始化一个空列表来保存结果
results_ori = []
results_ft = []

# 打开并读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 调整后的正则表达式
pattern = r"\[{'type': 'image', 'image': '(.*?)'}, {'type':"
image_info = re.findall(pattern, content)

print('路径：',image_info)

# 输出提取的图片路径
print(len(image_info))
# 使用正则表达式匹配 ori: 和 ft: 后面的 yes 或 no
ori_matches = re.findall(r'ori:\s*(yes|no)', content)
ft_matches = re.findall(r'ft:\s*(yes|no)', content)

# 合并 ori 和 ft 的结果到一个列表
for ori, ft in zip(ori_matches, ft_matches):
    # 将 ori 和 ft 的结果添加到结果列表
    results_ori.append(ori)
    results_ft.append(ft)

# 输出最终的结果
print('微调前：',results_ori)
print('微调后：',results_ft)
test_labels.pop(0)
print(len(test_labels))
print(len(results_ori))
print(len(results_ft))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 微调前的性能
accuracy_ori = accuracy_score(test_labels, results_ori)
precision_ori = precision_score(test_labels, results_ori, pos_label='yes')
recall_ori = recall_score(test_labels, results_ori, pos_label='yes')
f1_ori = f1_score(test_labels, results_ori, pos_label='yes')

# 微调后的性能
accuracy_ft = accuracy_score(test_labels, results_ft)
precision_ft = precision_score(test_labels, results_ft, pos_label='yes')
recall_ft = recall_score(test_labels, results_ft, pos_label='yes')
f1_ft = f1_score(test_labels, results_ft, pos_label='yes')

# 打印结果
print("===== 微调前模型性能 =====")
print(f"Accuracy: {accuracy_ori * 100:.2f}%")
print(f"Precision: {precision_ori * 100:.2f}%")
print(f"Recall: {recall_ori * 100:.2f}%")
print(f"F1-score: {f1_ori * 100:.2f}%")

print("\n===== 微调后模型性能 =====")
print(f"Accuracy: {accuracy_ft * 100:.2f}%")
print(f"Precision: {precision_ft * 100:.2f}%")
print(f"Recall: {recall_ft * 100:.2f}%")
print(f"F1-score: {f1_ft * 100:.2f}%")