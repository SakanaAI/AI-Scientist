import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'Arial'

# 图形与子图布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图数据：pairwise准确率
left_labels = ['Proposal', 'Paper']
x = np.arange(len(left_labels))

# 每组两个柱子：我们的方法 vs baseline
proposal_scores = [77.4, 71.0]   # Ours, AI-Researcher
paper_scores = [81.0, 64.0]      # Ours, DeepReview
left_ours = [proposal_scores[0], paper_scores[0]]
left_baseline = [proposal_scores[1], paper_scores[1]]

width = 0.35
colors = ['#00BFFF', '#FF6B6B']  # Ours蓝色，Baseline红色

# 左图绘制
ax1.bar(x - width/2, left_baseline, width, label='Baseline', color=colors[1])
ax1.bar(x + width/2, left_ours, width, label='Ours', color=colors[0])

ax1.set_ylabel('Pairwise Accuracy (%)')
ax1.set_title('Pairwise Evaluation Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(left_labels)
ax1.set_ylim(0, 100)
ax1.legend(loc='upper left')

# 添加数值标签
for i in range(len(left_labels)):
    ax1.text(x[i] - width/2, left_baseline[i] + 1, f'{left_baseline[i]:.1f}%', ha='center')
    ax1.text(x[i] + width/2, left_ours[i] + 1, f'{left_ours[i]:.1f}%', ha='center')

# 右图数据：refine pipeline带来的提升比例
right_labels = ['Proposal', 'Paper']
improve_rates = [80, 100]  # 改进百分比
bar_colors = ['#8BC34A', '#00BFFF']

ax2.bar(right_labels, improve_rates, color=bar_colors)
ax2.set_ylabel('Improvement Rate (%)')
ax2.set_title('Performance Improvement via Refine Pipeline')
ax2.set_ylim(0, 110)

# 添加数值标签
for i, v in enumerate(improve_rates):
    ax2.text(i, v + 2, f'{v}%', ha='center')

# 布局调整与保存
plt.tight_layout()
plt.savefig('pairwise_and_improvement.png', dpi=300, bbox_inches='tight')
# plt.show()  # 可打开查看
