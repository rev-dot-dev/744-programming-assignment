from pathlib import Path
from scipy import sparse
import numpy as np
import json
from tfidf import TFIDF_OUTPUT_PATH
import csv
import sys
from word2vec import EMBEDDING_OUTPUT_PATH
import data_preprocessing
import similarities
import evals

DATA_DIR = Path("PMC-Patients-full-v2")
TOP5_OUTPUT_PATH = Path("top5s_full.json")
EVALS_OUTPUT_PATH = Path("evals_full.json")

patient_dict = {}
top5_dict = {}
evals_dict = {}

# Compute word2vec document embeddngs

if __name__ == "__main__":
    dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR

    print("Loading patient data...")
    patient_dict, top5_dict, evals_dict = data_preprocessing.load_patient_data(dir)

    print("Loading tfidf matrix...")
    tfidf_matrix = sparse.load_npz(TFIDF_OUTPUT_PATH)

    print("Loading word2vec embeddings...")
    embeddings = np.array(list(csv.reader(open(EMBEDDING_OUTPUT_PATH, 'r'))))

    """
    print("Generating top 5 similar patients for each patient...")
    p_uids = list(patient_dict.keys())
    for idx, patient in enumerate(tqdm.tqdm(patient_dict.values())):
        patient_uid = patient['uid']
        # Get the similarity scores for the current patient
        tfidf_similarities = similarities.cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)[0]
        word2vec_similarities = similarities.cosine_similarity(embeddings[idx:idx+1], embeddings)[0]

        # Get the indices of the top 5 most similar patients
        top5_tfidf = similarities.get_top_5(patient_uid, p_uids, tfidf_similarities)
        top5_word2vec = similarities.get_top_5(patient_uid, p_uids, word2vec_similarities)

        # Store the results
        top5_dict[patient_uid]['tfidf'] = top5_tfidf
        top5_dict[patient_uid]['word2vec'] = top5_word2vec
    """

    p_uids = list(patient_dict.keys())

    print("Generating Word2Vec top 5")
    embeddings = embeddings.astype(np.float32)
    top5_w2v_dict = similarities.find_top_5_batch(embeddings, embeddings, p_uids, batch_size=500)

    print("Generating TF-IDF top 5")
    tfidf_matrix = tfidf_matrix.astype(np.float32)
    top5_tfidf_dict = similarities.find_top_5_batch(tfidf_matrix, tfidf_matrix, p_uids, batch_size=500)

    for uid in p_uids:
        top5_dict[uid]['tfidf'] = top5_tfidf_dict[uid]
        top5_dict[uid]['word2vec'] = top5_w2v_dict[uid]
        
    print("Saving top 5 similar patients to JSON...")
    with open(TOP5_OUTPUT_PATH, 'w') as f:
        json.dump(top5_dict, f)

    print("Evaluating the results...")
    for uid, data in top5_dict.items():
        similar_patients = patient_dict[uid]['similars']
        evals_dict[uid]['tfidf'] = {
            'precision_5': evals.get_precision_5(data['tfidf'], similar_patients),
            'recall_5': evals.get_recall_5(data['tfidf'], similar_patients)
        }
        evals_dict[uid]['word2vec'] = {
            'precision_5': evals.get_precision_5(data['word2vec'], similar_patients),
            'recall_5': evals.get_recall_5(data['word2vec'], similar_patients)
        }

    print("Saving evaluation results to JSON...")
    with open(EVALS_OUTPUT_PATH, 'w') as f:
        json.dump(evals_dict, f)

    


