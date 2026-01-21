import requests
import os


url = os.environ.get("NOVA_INDEX_API_URL")

def search_papers(query, topk=10):
    payload = {
        "request_id": "a13384c047994e4d8af7132082672545",
        "query": query,
        "use_reranker": False,
        "topK": topk,
        "fields": ["arxiv_id", "title", "abstract_info", "authors", "date"]
    }
    headers = {"content-type": "application/json"}
    response = requests.request("POST", url, json=payload, headers=headers)
    # print(response.text)
    papers = response.json()['paper_info']
    # import pdb;pdb.set_trace()
    for i in range(len(papers)):
        papers[i]['abstract'] = papers[i]['abstract_info'].replace('\n', ' ')
        papers[i]['title'] = papers[i]['title'].replace('\n', ' ')
        print('='*20)
        print("paper title:", papers[i]['title'])
    return papers


if __name__ == "__main__":
    query = "use llm generate novel research idea"
    results = search_papers(query)
    print(results[0])