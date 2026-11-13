import nltk
import os
from pathlib import Path
import json

DATA_DIR = Path("PMC_Patients")
OUTPUT_PATH = Path("preprocessed_files.json")
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

"""
preprocess_string: str -> [str]

Word tokenizes, removes punctuation and stop words, then
lemmatizes the provided string 
(intended for sentence/paragraph/document, but works on words too)
"""
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

if __name__ == "__main__":
    output_dict = {}

    # Iterate through .json files in 'PMC_Patients'
    for patient_file in DATA_DIR.iterdir():
        with open(patient_file, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)

            # Extract title, patient, combine text...
            # Could use some system to weight title differently
            # since there is higher density of keywords there, but not for now
            patient_text = ' '.join([patient_data['title'], patient_data['patient']])

            patient_preprocessed = preprocess_string(patient_text)

            # Store in dictionary to output as json
            output_dict[str(patient_file)] = patient_preprocessed

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        json.dump(output_dict, out)

