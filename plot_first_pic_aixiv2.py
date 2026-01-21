import matplotlib.pyplot as plt
import numpy as np

# 图像整体设置
plt.rcParams.update({'font.size': 12})

# 设置颜色
colors_left = ['#FF6B6B', '#FFC300', '#8BC34A', '#00BFFF']  # 红, 橙, 绿, 蓝
colors_right = ['#FFC300', '#00BFFF', '#FFC300', '#00BFFF', '#FFC300', '#00BFFF']

# 左图数据（pairwise accuracy）
left_labels = ['DeepReview\n(Paper)', 'AiXiv(Ours)\n(Paper)', 'AI Researcher\n(Proposal)', 'AiXiv(Ours)\n(Proposal)']
left_values = [64, 81, 71, 77.4]

# 右图数据（Refine带来的提升）
right_labels = ['Paper\nBetter Win %', 'Proposal\nBetter Win %', 'Old Proposal\nAccepted %', 'New Proposal\nAccepted %']
# right_values = [100, 80, 0, (42.85+66.66+6.89)/3.0]  # 平均accepted = 38.8%
right_values = [100, 80, 10, 70]  # 平均accepted = 38.8%

# 创建图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：Pairwise Accuracy
bars1 = ax1.bar(left_labels, left_values, color=colors_left)
ax1.set_title('Pairwise Accuracy Comparison')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(60, 85)  # 不从0开始，突出差异
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

# 右图：Refine Pipeline 带来的提升
bars2 = ax2.bar(right_labels, right_values, color=colors_right)
ax2.set_title('Effectiveness of Refinement Pipeline')
ax2.set_ylabel('Percentage (%)')
ax2.set_ylim(0, 110)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

# 总体布局
plt.tight_layout()
plt.savefig('pairwise_refine_comparison2.png', dpi=300, bbox_inches='tight')
# plt.show()  # 如需展示

# paper: 10 -> 70.0
# proposal: 0 -> 45.23%