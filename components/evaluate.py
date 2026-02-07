import datetime
import json
from collections import OrderedDict
from typing import Dict
import os
from pathlib import Path
import numpy as np
import ray
import ray.train.torch
import typer
from ray.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function
from typing_extensions import Annotated

from components import predict,utils,data
from components.config import logger
from components.predict import TorchPredictor

app = typer.Typer()

def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Get the overall performance metrics

    Args:
        Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        Dict: overall metrics.
    """

    metrics =  precision_recall_fscore_support(y_true,y_pred, average= "weighted")
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true))
    }

    return overall_metrics


def get_per_class_metrics(y_true: np.ndarray,y_pred: np.ndarray,class_to_index: Dict) -> Dict:
    """
    Get per class performance metrics

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        class_to_index (Dict): dictionary mapping class to index.

    Returns:
        Dict: per class metrics.
    """

    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true,y_pred,average= None)
    for i, _class in enumerate(class_to_index):
        per_class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    sorted_per_class_metrics = OrderedDict(sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True))
    return sorted_per_class_metrics


@slicing_function()
def nlp_llm(x):
    """NLP projects that use LLMs"""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return nlp_project and llm_project

@slicing_function()
def short_text(x):
    """Projects with short title and description"""
    return len(x.text.split()) < 8


def get_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, ds:Dataset) -> Dict:
    """
    Get performance metrics for slices

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        ds (Dataset): Ray dataset with labels.
    Returns:
        Dict: performance metrics for slices.
    """

    slice_metrics = {}
    df = ds.to_pandas()
    df["text"] = df["title"] + " " + df["description"]
    slices = PandasSFApplier([nlp_llm, short_text]).apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro")
            slice_metrics[slice_name] = {}
            slice_metrics[slice_name]["precision"] = metrics[0]
            slice_metrics[slice_name]["recall"] = metrics[1]
            slice_metrics[slice_name]["f1"] = metrics[2]
            slice_metrics[slice_name]["num_samples"] = len(y_true[mask])
    return slice_metrics


@app.command()
def evaluate(
    run_id: str = typer.Option(..., help="id of the specific run to load from"),
    dataset_loc: str = typer.Option(..., help="dataset (with labels) to evaluate on"),
    results_fp: str = typer.Option(..., help = "location to save evaluation results to")
) -> Dict:
    """
    Evaluates on the holdout dataset

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        dataset_loc (str): dataset (with labels) to evaluate on.
        results_fp (str, optional): location to save evaluation results to. Defaults to None.

    Returns:
        Dict: model's performance metrics on the dataset.
    """
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        num_gpus=0,              # 🔥 CRITICAL for Windows
        include_dashboard=True,
        dashboard_host="127.0.0.1",
        dashboard_port=8265,
        runtime_env={
            "env_vars": {
                "GITHUB_USERNAME": os.environ.get("GITHUB_USERNAME", "")
            }
        },
    )


    ds = data.load_data(dataset_loc)
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # y_true
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    values = preprocessed_ds.select_columns(cols = ["targets"]).take_all()
    y_true = np.stack([item["targets"] for item in values])

    # y_pred
    predictions = preprocessed_ds.map_batches(predictor).take_all()
    y_pred = np.array([d["output"] for d in predictions])

    # Metrics
    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(y_true=y_true, y_pred=y_pred, class_to_index=preprocessor.class_to_index),
        "slices": get_slice_metrics(y_true=y_true, y_pred=y_pred, ds=ds),
    }
    logger.info(json.dumps(metrics, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":  # pragma: no cover, checked during evaluation workload
    app()
