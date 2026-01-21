import random
import requests
from sklearn.cluster import KMeans

def get_embeddings(text_list, url='http://localhost:5000/compute_embedding'):
    response = requests.post(url, json={'texts': text_list})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error in getting embeddings")

def cluster_texts(text_list, embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    cluster_dict = {i: [] for i in range(k)}
    for idx, text, label in zip(list(range(len(text_list))), text_list, labels):
        cluster_dict[label].append([text, idx])
    return cluster_dict

def select_from_cluster(cluster_dict, cluster_index, select_all=False):
    if cluster_index not in cluster_dict:
        raise ValueError(f"聚类中心 {cluster_index} 不存在")

    cluster = cluster_dict[cluster_index]
    if not cluster:
        raise ValueError(f"聚类中心 {cluster_index} 为空")

    if select_all:
        return cluster
    else:
        return [random.choice(cluster)]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--URL", type=str, required=True)
    args = parser.parse_args()

    text_list = ["I love this movie", "I like this book", "I am a student"]
    embeddings = get_embeddings(text_list, args.URL)
    print(embeddings)

    """
    python cluster_tool.py --URL http://localhost:10041/compute_embedding
    """