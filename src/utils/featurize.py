import os
import logging
import pandas as pd
import joblib
import scipy.sparse as sparse
import numpy as np

def save_matrix(df, matrix, out_path):
    """
    Save the train matrix to a file.
    """
    logging.info("Saving train matrix to file...")
    

    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format='csr')

    msg = f"Saving train matrix to file... at {out_path} of size {result.shape} and type as {result.dtype}.\n"
    logging.info(msg)
    
    joblib.dump(result, out_path)

    logging.info("Saved train matrix to file.\n")
