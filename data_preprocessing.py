import nltk
import os
from pathlib import Path
import json

DATA_DIR = Path("PMC-Patients-full-v2")
OUTPUT_PATH = Path("preprocessed_files.json")
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

def preprocess_string(str):
    # Remove punctuation with regex, tokenize into words
    regexp_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = regexp_tokenizer.tokenize(str.lower())

    # Remove stop words with a given dictionary
    unstopped = [token for token in tokens if token not in STOP_WORDS]

    # Lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in unstopped]

    return lemmatized

# Load the patient data
def load_patient_data(dir):
    patient_dict = {}
    top5_dict = {}
    evals_dict = {}

    for patient_file in dir.iterdir():
        with open(patient_file, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
            patient_text = patient_data['title'] + ' ' + patient_data['patient']
            patient_preprocessed = ' '.join(preprocess_string(patient_text))
            patient_dict[patient_data['patient_uid']] = {
                'uid': patient_data['patient_uid'],
                'id': patient_data['patient_id'],
                'text': patient_preprocessed,
                'similars': json.loads(patient_data['similar_patients'].replace("'", '"'))
            }
            top5_dict[patient_data['patient_uid']] = {
                'id': patient_data['patient_id'],
                'tfidf': [],
                'word2vec': []
            }
            evals_dict[patient_data['patient_uid']] = {
                'id': patient_data['patient_id'],
                'tfidf': [],
                'word2vec': []
            }

    return patient_dict, top5_dict, evals_dict

