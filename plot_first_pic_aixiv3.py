import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams.update({'font.size': 12})

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

## 左图：Pairwise Accuracy 比较
labels_left = ['Proposal', 'Paper']
baseline_scores = [71.0, 64.0]      # Ai-Researcher, DeepReview
ours_scores = [77.4, 81.0]          # Ours
x = np.arange(len(labels_left))
width = 0.35

ax1.bar(x - width/2, baseline_scores, width, label='Baseline', color='#FF6B6B')  # Red for baseline
ax1.bar(x + width/2, ours_scores, width, label='Ours (aiXiv)', color='#4285F4')  # Blue for ours
ax1.set_ylabel('Pairwise Accuracy (%)')
ax1.set_title('Pairwise Evaluation Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_left)
ax1.set_ylim(60, 85)
ax1.legend()

for i in range(len(labels_left)):
    ax1.text(x[i] - width/2, baseline_scores[i] + 0.8, f'{baseline_scores[i]}%', ha='center')
    ax1.text(x[i] + width/2, ours_scores[i] + 0.8, f'{ours_scores[i]}%', ha='center')


## 右图：改进率 & Accepted Rate
labels_right = ['Improved Rate', 'Accepted Rate']
subcategories = ['Paper', 'Proposal']
improved = [100, 80]
accepted_old = [10, 0]
accepted_new = [70.0, 45.23]

# paper: 10 -> 70.0
# proposal: 0 -> 45.23%

x = np.arange(len(subcategories))
bar_width = 0.25

# 第一组：Improved Rate
ax2.bar(x - bar_width, improved, width=bar_width, color='#34A853', label='Improved (%)')  # Green for improvement

# 第二组：Old Accepted Rate
ax2.bar(x, accepted_old, width=bar_width, color='#FF6B6B', label='Old Accepted (%)')  # Red for old rate

# 第三组：New Accepted Rate
ax2.bar(x + bar_width, accepted_new, width=bar_width, color='#4285F4', label='New Accepted (%)')  # Blue for new rate

ax2.set_ylabel('Percentage (%)')
ax2.set_title('Refinement Effect: Improvement & Acceptance')
ax2.set_xticks(x)
ax2.set_xticklabels(subcategories)
ax2.set_ylim(0, 110)
ax2.legend()

# 添加文字
for i in range(len(x)):
    ax2.text(x[i] - bar_width, improved[i] + 2, f'{improved[i]}%', ha='center', fontsize=10)
    ax2.text(x[i], accepted_old[i] + 2, f'{accepted_old[i]}%', ha='center', fontsize=10)
    ax2.text(x[i] + bar_width, accepted_new[i] + 2, f'{accepted_new[i]:.1f}%', ha='center', fontsize=10)

# 布局优化
plt.tight_layout()
plt.savefig('pairwise_refine_fullv3.png', dpi=300)
# plt.show()
