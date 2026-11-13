import nltk
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from pandas import DataFrame
from data_preprocessing import preprocess_string
import csv

DATA_DIR = Path("PMC_Patients")
OUTPUT_PATH = Path("tfidf.csv")
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

def process_documents(dir):
    documents = []
    doc_names = []
    
    # Iterate through .json files in 'PMC_Patients'
    for patient_file in sorted(DATA_DIR.iterdir()):
        with open(patient_file, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)

            # Extract title, patient, combine text...
            # Could use some system to weight title differently
            # since there is higher density of keywords there, but not for now
            patient_text = ' '.join([patient_data['title'], patient_data['patient']])

            # Process and convert back to single document string
            patient_preprocessed = ' '.join(preprocess_string(patient_text))

            documents.append(patient_preprocessed)
            doc_names.append(patient_file.stem)

    return documents, doc_names

def construct_tfidf_matrix(documents):  
    # Sklearn's tfidf vectorizer expects an iterable of strings
    # each of which is the entire document to be processed.
    # Now that they are in the documents list, pass to function
    
    return tfidf_matrix

if __name__ == "__main__":
    documents, doc_names = process_documents(DATA_DIR)
    # sklearn returns this as a sparse matrix representation
    # of (doc_id, term_id): tfidf_value
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    # sparse.save_npz('sparse_tfidf.npz', sparse_matrix)

    tfidf_df = DataFrame(tfidf_matrix.toarray(), index=doc_names, columns=vectorizer.get_feature_names_out())
    tfidf_df.to_csv(OUTPUT_PATH, index=False)

    #with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as out:
    #    writer = csv.writer(out)
    #    writer.writerow(vectorizer.get_feature_names_out())
    #    writer.writerows(tfidf_matrix.toarray())
    #print(vectorizer.get_feature_names_out())
    #print(tfidf_array)
    
