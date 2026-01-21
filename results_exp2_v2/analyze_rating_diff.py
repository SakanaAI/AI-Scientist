import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt



"""
1. 读取v1版本rating结果:ai_sci_deepreview_results.json，为一个列表，每个key代表每篇文章的title，value包括meta review的结果。例如 data[key][0]['meta_review']为meta review的结果，还包括这几个key：                "rating": 3.0,
                "soundness": "2.25",
                "presentation": "1.5",
                "contribution": "1.5",
2. 读取v2版本rating结果：ai_sci_new_paperdeepreview_results_v22.json，为一个列表，每个key代表每篇文章的title，value包括新版本paper的meta review的结果。
    - meta review包括整体rating分数。以及soundness, presentation, contribution三个维度的评分。例如：                "rating": 2.5, "soundness": "2.0", "presentation": "1.75", "contribution": "1.75",
3. 分析v1和v2版本meta review前后的变化, 说明其变化趋势。
    - 例如rating的G:S:B数据 (如果v2>v1则G+1, 如果v2==v1,则S+1, 如果v2<v1,则B+1)
    - 例如tating的平均值变化
    - 其他分析结果
4. 绘出漂亮的学术图，保留分析结果到:结果分析/figures文件夹下
6. 将每篇文章的前后的变化的结果保留到:结果分析/csv文件中

"""



def load_json_data(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_ratings(v1_data: Dict, v2_data: Dict) -> Tuple[Dict, Dict, Dict]:
    # Initialize counters
    gsb_counts = {'G': 0, 'S': 0, 'B': 0}
    avg_changes = {'rating': 0, 'soundness': 0, 'presentation': 0, 'contribution': 0}
    paper_changes = {}

    # paper, rating_v1, rating_v2, soundness_v1, soundness_v2, presentation_v1, presentation_v2, contribution_v1, contribution_v2
    paper_changes_list = []

    for title in v1_data.keys():
        v2_key = title + '_improved_v2'

        assert v2_key in v2_data, f"{v2_key} not in v2_data, v2_data.keys(): {v2_data.keys()}"

        v1_ratings = v1_data[title][0]['meta_review']
        v2_ratings = v2_data[v2_key][0]['meta_review']

        # Calculate G:S:B
        if float(v2_ratings['rating']) > float(v1_ratings['rating']):
            gsb_counts['G'] += 1
        elif float(v2_ratings['rating']) == float(v1_ratings['rating']):
            gsb_counts['S'] += 1
        else:
            gsb_counts['B'] += 1

        # Calculate changes for each metric
        paper_changes[title] = {}
        for metric in ['rating', 'soundness', 'presentation', 'contribution']:
            change = float(v2_ratings[metric]) - float(v1_ratings[metric])
            avg_changes[metric] += change
            paper_changes[title][metric] = change

        # Add to paper_changes_list
        paper_changes_list.append([
            title,
            v1_ratings['rating'],
            v2_ratings['rating'],
            v1_ratings['soundness'],
            v2_ratings['soundness'],
            v1_ratings['presentation'],
            v2_ratings['presentation'],
            v1_ratings['contribution'],
            v2_ratings['contribution']
        ])

    # Calculate averages
    n_papers = len(paper_changes)
    for metric in avg_changes:
        avg_changes[metric] /= n_papers

    return gsb_counts, avg_changes, paper_changes, paper_changes_list

def plot_results(gsb_counts: Dict, avg_changes: Dict, paper_changes_list: List, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Plot GSB distribution in academic style
    plt.figure(figsize=(10, 4))
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Professional color scheme
    x = np.arange(1)
    width = 0.25

    plt.bar(x - width, [gsb_counts['G']], width, label='Good (V2 > V1)', color=colors[0])
    plt.bar(x, [gsb_counts['S']], width, label='Same (V2 = V1)', color=colors[1])
    plt.bar(x + width, [gsb_counts['B']], width, label='Bad (V2 < V1)', color=colors[2])

    plt.title('Distribution of Rating Changes Between Versions', pad=15)
    plt.ylabel('Number of Papers')
    plt.xticks([])  # Remove x-axis ticks
    plt.legend(frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value labels on top of bars
    for i, v in enumerate([gsb_counts['G'], gsb_counts['S'], gsb_counts['B']]):
        plt.text(x - width + i*width, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gsb_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot average changes with academic style
    plt.figure(figsize=(12, 6))
    metrics = list(avg_changes.keys())

    # Calculate average values for each metric
    v1_values = []
    v2_values = []
    for paper in paper_changes_list:
        v1_values.append([paper[1], float(paper[3]), float(paper[5]), float(paper[7])])
        v2_values.append([paper[2], float(paper[4]), float(paper[6]), float(paper[8])])

    v1_means = np.mean(v1_values, axis=0)
    v2_means = np.mean(v2_values, axis=0)

    x = np.arange(len(metrics))
    width = 0.35

    # Use professional color scheme
    color_v1 = '#3498db'  # Blue
    color_v2 = '#2ecc71'  # Green

    # Create grouped bars
    plt.bar(x - width/2, v1_means, width, label='Version 1', color=color_v1, alpha=0.8)
    plt.bar(x + width/2, v2_means, width, label='Version 2', color=color_v2, alpha=0.8)

    plt.title('Comparison of Metrics Between Versions', pad=15, fontsize=12)
    plt.ylabel('Average Score', fontsize=11)
    plt.xlabel('Metrics', fontsize=11)
    plt.xticks(x, metrics, rotation=45)
    plt.legend(frameon=False)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value labels on top of bars
    for i in range(len(metrics)):
        plt.text(i - width/2, v1_means[i], f'{v1_means[i]:.2f}',
                ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, v2_means[i], f'{v2_means[i]:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_to_csv(paper_changes: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Prepare CSV content
    csv_content = ['Title,Rating Change,Soundness Change,Presentation Change,Contribution Change']
    for title, changes in paper_changes.items():
        row = [
            title,
            str(changes['rating']),
            str(changes['soundness']),
            str(changes['presentation']),
            str(changes['contribution'])
        ]
        csv_content.append(','.join(row))

    # Save to file
    with open(os.path.join(save_dir, 'rating_changes.csv'), 'w') as f:
        f.write('\n'.join(csv_content))


def save_paper_changes_list(paper_changes_list: List, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # use pandas convert to df and Save to file
    df = pd.DataFrame(paper_changes_list, columns=['Title', 'Rating V1', 'Rating V2', 'Soundness V1', 'Soundness V2',
              'Presentation V1', 'Presentation V2', 'Contribution V1', 'Contribution V2'])
    df.to_csv(os.path.join(save_dir, 'paper_changes_list.csv'), index=False)


def main():
    # Load data
    v1_data = load_json_data('ai_sci_deepreview_results.json')
    v2_data = load_json_data('ai_sci_new_paperdeepreview_results_v22.json')

    # Analyze ratings
    gsb_counts, avg_changes, paper_changes, paper_changes_list = analyze_ratings(v1_data, v2_data)

    # Create output directories
    figures_dir = os.path.join('结果分析', 'figures')
    csv_dir = os.path.join('结果分析', 'csv')

    # Generate plots and save results
    plot_results(gsb_counts, avg_changes, paper_changes_list, figures_dir)
    save_to_csv(paper_changes, csv_dir)
    save_paper_changes_list(paper_changes_list, csv_dir)

    # Print summary
    print("Analysis completed:")
    print(f"GSB Distribution: {gsb_counts}")
    print(f"Average Changes: {avg_changes}")

if __name__ == "__main__":
    main()


