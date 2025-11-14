# 假设已知的数据
accuracy = 57.06  # Accuracy (%)
precision = 48.87  # Precision (%)
recall = 47.48  # Recall (%)

# 计算 F1-score
f1_score = 2 * (precision * recall) / (precision + recall)

# 假设 mAP 可以用 Precision 作为近似（通常是有多个类别时才计算）
# 如果是二分类任务，mAP 近似于 Precision
map_score = precision

# 输出结果
print(f"F1-score (%): {f1_score:.2f}")
print(f"mAP (%): {map_score:.2f}")