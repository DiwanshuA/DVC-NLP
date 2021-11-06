import argparse
import os
import shutil
from tqdm import tqdm
import logging
import joblib
import numpy as np
from sklearn.metrics import metrics
from src.utils import read_yaml, save_json
import math

STAGE = "Four"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def main(config_path):
    config = read_yaml(config_path)

    artifacts = config["artifacts"]
    featurized_data_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA_DIR"])
    featurized_test_data_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_OUT_TEST"])

    # Load the model
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    model = joblib.load(model_path)
    
    matrix = joblib.load(featurized_test_data_path)

    X = matrix[:, 2]
    labels = np.squeeze(matrix[:, 1]).toarray()

    # Predict
    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]
    #predictions = np.argmax(predictions_by_class, axis=1)

    PRC_json_path = config["plots"]["PRC"]
    ROC_json_path = config["plots"]["ROC"]
    scores_json_path = config["plots"]["SCORES"]

    avg_prc = metrics.average_precision_score(labels, predictions)
    roc_auc = metrics.roc_auc_score(labels, predictions)

    scores = {
        "avg_prc": avg_prc,
        "roc_auc": roc_auc
    }

    save_json(scores_json_path, scores)

    precision, recall, prc_threshold = metrics.precision_recall_curve(labels, predictions)

    nth_point = int(math.ceil(len(prc_threshold) * 0.0001))
    prc_points = list(zip(recall, precision, prc_threshold))[::nth_point]

    prc_data = {
        "prc": [
            {
                "recall": r,
                "precision": p,
                "threshold": t
            } for p, r, t in prc_points
        ]
    }

    save_json(PRC_json_path, prc_points)

    fpr, tpr, roc_threshold = metrics.roc_curve(labels, predictions)

    roc_data = {
        "roc": [
            {
                "fpr": fp,
                "tpr": tp,
                "threshold": t
            } for fp, tp, t in zip(fpr, tpr, roc_threshold)
        ]
    }

    save_json(ROC_json_path, roc_data)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>>Stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>>Stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e