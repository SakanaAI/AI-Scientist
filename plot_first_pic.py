"""
参考下面代码完成一个图，给出python代码。

左图：
pairwise的准确率，包括proposal的和paper的pairwise的准确率。


paper pairwise的准确率:
对比：
deepreview pairwise的准确率：64%
aixiv(Our): 81%


proposal pairwise的准确率：77.4%
Ai-Researhcer的pairwise的准确率：71%


右图：
包括我们refine pipeline带来的提升：

proposal:提升的比例：80%
paper的提升：100%

请给出python代码，给出的代码要和图的内容一致。

同时给出改图的caption latex文字：主题包括。1. 我们pairwise的方法在paper/proposal的评估上都有sota效果。能验证proposal和paper修改前后是否有提升。2. 实验表明我们提出的reivew的方法，能显著给proposal,paper带来提升。
"""



import matplotlib.pyplot as plt
import numpy as np

# 数据准备
# 左侧图表数据
categories = ['Research agent', 'AI researcher', 'AI scientist', 'Nova(Ours)']
top_scores = [223, 237, 249, 619]
colors_left = ['#FF6B6B', '#FFC300', '#8BC34A', '#00BFFF']  # 颜色顺序：红色、橙色、绿色、蓝色

# 右侧图表数据
iterations = [0, 1, 2, 3]
unique_ideas = [14.7, 26.9, 41.3, 50.6]
error_bars = [2, 3, 4, 5]  # 假设误差棒的大小
colors_right = ['#FF6B6B', '#FFC300', '#8BC34A', '#00BFFF']  # 颜色顺序：红色、橙色、绿色、蓝色

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制左侧图表
ax1.bar(categories, top_scores, color=colors_left)
ax1.set_xlabel('Category')
ax1.set_ylabel('# of top scored ideas')
ax1.set_ylim(0, 650)  # 设置 y 轴范围
ax1.set_title('# of top scored ideas')

# 添加顶部数值标签
for i, v in enumerate(top_scores):
    ax1.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=10)

# 绘制右侧图表
ax2.bar(iterations, unique_ideas, color=colors_right, yerr=error_bars, capsize=5)
ax2.set_xlabel('Iteration step')
ax2.set_ylabel('# of unique novel ideas')
ax2.set_ylim(0, 65)  # 设置 y 轴范围
ax2.set_title('# of unique novel ideas')

# 添加顶部数值标签
for i, v in enumerate(unique_ideas):
    ax2.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图表
# plt.show()

plt.savefig('first_pic.png', bbox_inches='tight', dpi=300)