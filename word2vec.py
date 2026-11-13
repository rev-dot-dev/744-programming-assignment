# from gensim.models import word2vec
from gensim.utils import tokenize
from gensim import downloader
from pathlib import Path
import numpy as np
import json
import csv

DATA_DIR = Path("PMC-Patients-full-v2")
EMBEDDING_OUTPUT_PATH = Path("word2vec_embeddings_full.csv")
word2vec_model = downloader.load('word2vec-google-news-300')

def compute_document_embedding(text):
    tokens = list(tokenize(text, lowercase=True))
    # Compute the average embedding for the document
    embeddings = np.array([word2vec_model[token] for token in tokens if token in word2vec_model])
    if embeddings.size > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def create_embeddings(patient_data):
    embeddings_array = []
    for patient_id, data in patient_data.items():
        
        text = data['text']
        embedding = compute_document_embedding(text)
        embeddings_array.append(embedding)

    return embeddings_array

# Could be batched for faster processing
def w2v_main(dir):
    files_written = 0
    with open(EMBEDDING_OUTPUT_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for patient_file in Path(dir).iterdir():
            with open(patient_file, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
                text = patient_data['title'] + " " + patient_data['patient']
                embedding = compute_document_embedding(text)
                writer.writerow(embedding)
                files_written += 1
                if files_written % 1000 == 0:
                    print(f"Processed {files_written} files...")
    print(f"Finished processing {files_written} patient files.")

if __name__ == "__main__":
    w2v_main(DATA_DIR)