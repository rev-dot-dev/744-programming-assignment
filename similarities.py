from pathlib import Path
import tqdm
import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse

# Returns a list of the top 5 patient uids
def get_top_5(patient_uid, patient_uids, similarity_vector):
    similarity_dict = dict(zip(patient_uids, similarity_vector))
    sorted_dict = sorted(similarity_dict, key=similarity_dict.get, reverse=True)
    sorted_dict.remove(patient_uid) # should be first element anyways
    return sorted_dict[:5]

# Returns a list of the top 5 pateint uids, executed batch_size at a time
def find_top_5_batch(query_matrix, doc_matrix, uids, batch_size=500):
    # Normalize upfront: cosine similarity is now just dot product
    doc_norm = normalize(doc_matrix, norm='l2', axis=1, copy=False)
    # Ensure float32 to halve memory and double matrix math speed
    if not sparse.issparse(doc_norm) and doc_norm.dtype != np.float32:
        doc_norm = doc_norm.astype(np.float32)
    
    # Pre-transpose docs for fast dot products
    doc_T = doc_norm.T
    n_queries = doc_norm.shape[0]
    uids_arr = np.array(uids)
    
    top5_results = {}
    
    for start in tqdm.tqdm(range(0, n_queries, batch_size)):
        end = min(start + batch_size, n_queries)
        batch = doc_norm[start:end]
        
        # Batch matrix multiplication: (batch_size, 100k) in float32 (~200MB RAM)
        sim_batch = batch.dot(doc_T)
        if sparse.issparse(sim_batch):
            sim_batch = sim_batch.toarray()
            
        # In-place top-6 selection (patient + top 5)
        # argpartition is O(N) in C — thousands of times faster than Python dict sorting
        top_k = 6
        partitioned_indices = np.argpartition(-sim_batch, range(top_k), axis=1)[:, :top_k]
        
        for i, row_indices in enumerate(partitioned_indices):
            global_idx = start + i
            query_uid = uids_arr[global_idx]
            
            # Sort just the top 6 elements by descending score
            scores = sim_batch[i, row_indices]
            sorted_order = np.argsort(-scores)
            sorted_candidates = row_indices[sorted_order]
            
            # Filter out self
            top_candidates = [idx for idx in sorted_candidates if idx != global_idx][:5]
            top5_results[query_uid] = uids_arr[top_candidates].tolist()
            
    return top5_results
    
