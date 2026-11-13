import json
from pathlib import Path

DATA_DIR = Path("PMC_Patients")
TOP5PATH = Path("top5s.json")
OUTPUT_PATH = Path("pr.json")

def get_precision_5(results, similar_patients):
    total = 5
    relevant = set(results).intersection(set(similar_patients.keys()))
    return len(relevant) / total

def get_recall_5(results, similar_patients):
    total = len(similar_patients)
    relevant = set(results).intersection(set(similar_patients.keys()))
    if total > 0:
        recall = len(relevant) / total
    else:
        recall = 1
    return  recall

if __name__ == "__main__":
    data = {}
    output = {}
    similar_patients = {}

    for patient_file in DATA_DIR.iterdir():
        with open(patient_file, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
            similar_patients[patient_data["patient_uid"]] = json.loads(patient_data['similar_patients'].replace("'", '"'))

    with open(TOP5PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for patient_uid, results in data.items():
        output[patient_uid] = {
            "patient_id": results["patient_id"],
            "tfidf": {
                "precision_5": get_precision_5(results["tfidf"], similar_patients[patient_uid]),
                "recall_5": get_recall_5(results["tfidf"], similar_patients[patient_uid]),
            },
            "word2vec": {
                "precision_5": get_precision_5(results["word2vec"], similar_patients[patient_uid]),
                "recall_5": get_recall_5(results["word2vec"], similar_patients[patient_uid]),
            }
        }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        json.dump(output, out)

    



