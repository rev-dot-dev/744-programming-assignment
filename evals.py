import json
from pathlib import Path

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

    



