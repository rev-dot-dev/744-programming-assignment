from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

DATA_DIR = Path("PMC_Patients")
TFIDF_MATRIX_FILEPATH = Path("tfidf.csv")
W2V_FILEPATH = Path("word2vec_embeddings.csv")
OUTPUT_PATH = Path("top5s.json")

# Returns a list of the top 5 patient uids
def get_top_5(patient_uid, patient_uids, similarity_vector):
    similarity_dict = dict(zip(patient_uids, similarity_vector))
    sorted_dict = sorted(similarity_dict, key=similarity_dict.get, reverse=True)
    sorted_dict.remove(patient_uid) # should be first element anyways
    return sorted_dict[:5]

if __name__ == "__main__":
    p_uids = []
    output = {}

    # Get UIDs and IDs
    for path in sorted(DATA_DIR.iterdir()):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            p_uids.append(data['patient_uid'])
            output[data['patient_uid']] = {"patient_id": data['patient_id'], "tfidf": [], "word2vec": []}
    
    tfidf_df = pd.read_csv(TFIDF_MATRIX_FILEPATH)
    tfidf_similarity_matrix = cosine_similarity(tfidf_df)
    del tfidf_df

    # Doing this one at a time to save memory reqs, huge csv files
    for i in range(len(p_uids)):
        top_5 = get_top_5(p_uids[i], p_uids, tfidf_similarity_matrix[i])
        output[p_uids[i]]["tfidf"] = top_5
    del tfidf_similarity_matrix

    word2vec_embeddings = pd.read_csv(W2V_FILEPATH, header=None)
    word2vec_similarity_matrix = cosine_similarity(word2vec_embeddings)
    del word2vec_embeddings

    for i in range(len(p_uids)):
        top_5 = get_top_5(p_uids[i], p_uids, word2vec_similarity_matrix[i])
        output[p_uids[i]]["word2vec"] = top_5 

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        json.dump(output, out)
    
