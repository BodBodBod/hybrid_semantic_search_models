import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from Retrievers import TransformerRetriever, LexicalRetriever, HybridScorer

# Recall@k Evaluation
def make_recall_function(transformer, transformer_embeds, queries, corpus_id_map, lexical_retriever, k=5):

    print(f"Precomputing scores for {transformer.model.config._name_or_path}...")

    all_transformer_scores = []
    all_lexical_scores = []

    for query in tqdm(queries, desc="Precomputing"):
        query_embed = transformer.get_query_embedding(query)
        transformer_scores = cosine_similarity(query_embed.reshape(1, -1), transformer_embeds)[0]
        lex_scores = lexical_retriever.get_query_scores(query)

        all_transformer_scores.append(transformer_scores)
        all_lexical_scores.append(lex_scores)

    def evaluate(alpha, beta, gamma, return_value = 'mean'):
        scorer = HybridScorer(alpha, beta, gamma)
        recall_scores = []

        for qid in range(len(queries)):
            transformer_scores = all_transformer_scores[qid]
            lex_scores = all_lexical_scores[qid]

            combined_scores = scorer.compute(
                lex_scores.get('bm25'),
                lex_scores.get('tfidf'),
                transformer_scores
            )
            topk_idx = combined_scores.argsort()[-k:][::-1]
            correct = sum(1 for idx in topk_idx if corpus_id_map[idx] == (qid, "pos"))
            recall_scores.append(correct / k)

        result = np.mean(recall_scores) if return_value == 'mean' else recall_scores
        return result

    return evaluate

# grid search weights alpha, beta, gamma
def grid_search_weights(recall_function, model_name, step=0.5, n_jobs=-1):

    results = []
    weight_range = np.arange(0, 1 + step, step)

    # triplets (alpha, beta, gamma)
    all_combinations = [
        (alpha, beta, 1.0 - alpha - beta)
        for alpha, beta in itertools.product(weight_range, repeat=2)
        if 0 <= (1.0 - alpha - beta) <= 1

    ]

    def process(alpha, beta, gamma):
        recall = recall_function(alpha, beta, gamma)
        return {
            "model": model_name,
            "alpha (BM25)": alpha,
            "beta (TF-IDF)": beta,
            "gamma (Transformer)": gamma,
            "recall@5": recall
        }

    print(f"Total combinations to evaluate: {len(all_combinations)}")

    # parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(alpha, beta, gamma) for alpha, beta, gamma in tqdm(all_combinations, desc=f"Grid search for {model_name}")
    )

    return pd.DataFrame(results)
