from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from data_preprocessing import load_patient_data
import tqdm

TFIDF_OUTPUT_PATH = Path("tfidf_full.npz")

def compute_tfidf_matrix(patient_dict):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(tqdm.tqdm([patient['text'] for patient in patient_dict.values()]))

def save_tfidf_matrix(tfidf_matrix, output_path):
    sparse.save_npz(output_path, tfidf_matrix)

def tfidf_main(patient_dict):
    tfidf_matrix = compute_tfidf_matrix(patient_dict)
    save_tfidf_matrix(tfidf_matrix, TFIDF_OUTPUT_PATH)
    return tfidf_matrix

if __name__ == "__main__":
    patient_dict, top5_dict, evals_dict = load_patient_data()
    tfidf_main(patient_dict)
    

    
