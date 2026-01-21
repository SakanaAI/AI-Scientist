import json
import shutil
import pandas as pd
import numpy as np
import os
import glob
import logging
import random
from typing import List
import argparse
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
from utils_tool import load_json_from_file, save_json_data_to_file, safe_mkdir, init_logging
from cluster_tool import cluster_texts, select_from_cluster, get_embeddings

logger = logging.getLogger()
random.seed(2024)




def get_dedup_results_from_initial_proposal_files(
        proposal_list,
        similarity_threshold=0.8,
        EMBEDING_SERVER_URL=None
    ):

    all_ideas = []
    for proposal in proposal_list:
        all_ideas.append(proposal["Experiment"])

    # Use the http interface to get the embedding vector
    embeddings = get_embeddings(all_ideas, EMBEDING_SERVER_URL)
    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    # Convert the similarity matrix to a numpy array
    ## setting the diagonal to 0
    np.fill_diagonal(similarity_matrix, 0)

    final_proposals = []
    final_ideas = []
    filter_idx = [] ## ideas that should be filtered
    non_duplicate_count = []
    non_duplicate_percentage = []
    filter_file_to_exist_file_map = dict()
    repeat_idea_map = dict()

    for i in range(len(all_ideas)):
        if i not in filter_idx:
            ## add current idea to filtered_ideas
            final_ideas.append([i, all_ideas[i], proposal_list[i]])
            final_proposals.append(proposal_list[i])
            ## filter out similar ideas
            for j in range(i+1, len(all_ideas)):
                if j not in filter_idx and similarity_matrix[i][j] > similarity_threshold or all_ideas[j] == all_ideas[i]:
                    filter_idx.append(j)
                    repeat_idea_map[all_ideas[j]] = [i, all_ideas[i], j]
                    print('\n')
                    print("==" * 20)
                    print(f"find similar idea\n\n idea1:id:{j}, {all_ideas[j]} \n\n\n idea2:{i}, {all_ideas[i]}")
        non_duplicate_count.append(len(final_ideas))
        non_duplicate_percentage.append(len(final_ideas) / (i + 1) * 100)

    print ("#final ideas: ", len(final_ideas))
    print("#non_duplicate_count: ", non_duplicate_count)
    print("#non_duplicate_percentage: ", non_duplicate_percentage)

    return final_proposals, final_ideas, repeat_idea_map



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal_list_file', type=str, default="", help='')
    parser.add_argument('--similarity_threshold', type=float, default=0.8, help='NN Similarity Threshold')
    parser.add_argument('--EMBEDING_SERVER_URL', type=str, default="", help='')
    parser.add_argument('--final_dedup_proposal_list_file', type=str, default="", help='')
    parser.add_argument('--output_dir', type=str, default="", help='')
    parser.add_argument('--topk_for_experiment', type=int, default=5, help='Topk for experiment')

    args = parser.parse_args()

    proposal_list = load_json_from_file(args.proposal_list_file)
    final_proposals, final_ideas, repeat_idea_map = get_dedup_results_from_initial_proposal_files(
        proposal_list,
        args.similarity_threshold,
        args.EMBEDING_SERVER_URL
    )
    save_json_data_to_file(final_proposals, args.final_dedup_proposal_list_file)
    print(f"final_dedup_proposal_list_file save to: {args.final_dedup_proposal_list_file}")



    # rank and select final ideas for experiment
    novel_ideas = [idea for idea in final_proposals if idea["novel"]]
    # novel_ideas = list(reversed(novel_ideas))
    print(f"Running {len(novel_ideas)} novel ideas")
    # rank and select topk idea according to Interestingness and Feasibility(th>=0.8) score
    novel_ideas =[_ for _ in novel_ideas if _["Feasibility"] >= 0.8]
    novel_ideas.sort(key=lambda x: x["Interestingness"] + x["Feasibility"], reverse=True)
    novel_ideas = novel_ideas[:args.topk_for_experiment]
    for i, idea in enumerate(novel_ideas):
        save_json_data_to_file([idea], args.output_dir + f'exp_idea_{i}.json')
        print(f"exp_idea_{i}.json save to: {args.output_dir + f'exp_idea_{i}.json'}")

    """
    python dedup_tool.py --proposal_list_file=templates/2d_diffusion/new_ideas.json --similarity_threshold=0.8 --EMBEDING_SERVER_URL=http://127.0.0.1:10041/compute_embedding --final_dedup_proposal_list_file=templates/2d_diffusion/final_dedup_proposals.json --output_dir=templates/2d_diffusion/ --topk_for_experiment 5
    python dedup_tool.py --proposal_list_file=templates/grokking/new_ideas.json --similarity_threshold=0.8 --EMBEDING_SERVER_URL=http://127.0.0.1:10041/compute_embedding --final_dedup_proposal_list_file=templates/grokking/final_dedup_proposals.json --output_dir=templates/grokking/ --topk_for_experiment 5
    python dedup_tool.py --proposal_list_file=templates/nanoGPT/new_ideas.json --similarity_threshold=0.8 --EMBEDING_SERVER_URL=http://127.0.0.1:10041/compute_embedding --final_dedup_proposal_list_file=templates/nanoGPT/final_dedup_proposals.json --output_dir=templates/nanoGPT/ --topk_for_experiment 5
    """