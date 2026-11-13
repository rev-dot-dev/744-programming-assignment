from gensim.models import word2vec
from gensim.utils import tokenize
from gensim import downloader
from pathlib import Path
import numpy as np
import json
import csv

DATA_DIR = Path("PMC_Patients")
OUTPUT_PATH = Path("word2vec_embeddings.csv")

embeddings_list = []
word2vec_model = downloader.load('word2vec-google-news-300')

for patient_file in sorted(DATA_DIR.iterdir()):
    with open(patient_file, 'r', encoding='utf-8') as f:
        patient_data = json.load(f)

        patient_text = ' '.join([patient_data['title'], patient_data['patient']])
        tokens = list(tokenize(patient_text, lowercase=True))

        patient_embedding = np.array([word2vec_model[token] for token in tokens if token in word2vec_model]).mean(axis=0)
        embeddings_list.append(patient_embedding)

with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerows(embeddings_list)




