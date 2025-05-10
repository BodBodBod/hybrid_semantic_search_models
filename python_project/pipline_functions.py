import numpy as np
import pandas as pd
import pickle
from Retrievers import TransformerRetriever, LexicalRetriever, HybridScorer
from model_quality import make_recall_function, grid_search_weights
from scipy.stats import wilcoxon

# serialize and load pipline
def serialize(model, model_name, corpus, queries, files_path):

    print(f"Caching corpus embeddings for {model_name}...")
    model.cache_corpus_embeddings(corpus)

    print(f"Caching query embeddings for {model_name}...")
    model.cache_query_embeddings(queries)

    model.save_embeddings(files_path + '/transformer_embeddings/' + model_name.split('/')[1] + '_embeddings.pkl')

    return model

def load(model, model_name, files_path):

    print(f"Loading corpus embeddings for {model_name}...")
    model.load_embeddings(files_path + '/transformer_embeddings/' + model_name.split('/')[1] + '_embeddings.pkl')

    return model

def ISL(model_name, files_path, corpus = False, queries = False, serialize_flg = False, load_flg = True):

    model = TransformerRetriever(model_name)

    if serialize_flg:
        print(f'Serializing {model_name}')
        if corpus and queries:
            model = serialize(model, model_name, corpus, queries, files_path)
            print(f'{model_name} embeddings are serialized\n')
        else:
            print('Initialize corpus and queries')

    if load_flg:
        model = load(model, model_name, files_path)
        print(f'{model_name} embeddings loaded\n')

    return model

# iterating through the parameters
grid = .02
all_jobs = -1

def calculate_load_grid_pipline(model_name, files_path, model_recall_at_k = None, grid = grid, n_jobs = all_jobs, calculate_flg = False, load_flg = True):

    if calculate_flg:
        print(f'Calculating results for {model_name}')
        results = grid_search_weights(
            model_recall_at_k,
            model_name=f"{model_name}",
            step=grid,
            n_jobs=all_jobs
        )

        results['model'] == model_name.split('/')[1]

        # save results
        results.to_csv(files_path + '/experiments_results/' + model_name.split('/')[1] + '_grid_search_result.csv', index = None)

    if load_flg:
        results = pd.read_csv(files_path + '/experiments_results/' + model_name.split('/')[1] + '_grid_search_result.csv')

    return results

# statistical comparasion using Wilcoxon signed-test
def stat_comparasion(result_df, recall_func, alpha = 0, beta = 0, gamma = 0):

    if alpha + beta + gamma != 1:
        best_solution = result_df.sort_values(by = 'recall@5', ascending = False).iloc[0]
        alpha, beta, gamma = best_solution['alpha (BM25)'], best_solution['beta (TF-IDF)'], best_solution['gamma (Transformer)']

    alpha_baseline, beta_baseline, gamma_baseline = 0, 0, 1 # transformer only

    if gamma != 1:
        stat_test_res = wilcoxon(
            x = recall_func(alpha, beta, gamma, return_value = 'vector'),
            y = recall_func(alpha_baseline, beta_baseline, gamma_baseline, return_value = 'vector')
          )
    else:
        stat_test_res = 'baseline and best solution are similar'

    return stat_test_res
