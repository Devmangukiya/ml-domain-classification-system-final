import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse
import os

import numpy as np
import ray
import typer
from numpyencoder import NumpyEncoder
from ray.air import Result
from ray.train.torch.torch_checkpoint import TorchCheckpoint
from typing_extensions import Annotated

from components.config import logger, mlflow
from components.data import CustomPreprocessor
from components.model import FinetunedLLM
from components.utils import collate_fn



## Initialize Typer CLI app
app = typer.Typer()

def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    """
    Decodes indices to labels

    Args:
        indices (Iterable[Any]): Iterable (list, array, etc.) with indices.
        index_to_class (Dict): mapping btw indices and labels

    Returns:
        List: list of labels
    """

    return [index_to_class[index] for index in indices]


def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    """Format probabilities to a dictionary mapping class label to probability.

    Args:
        prob (Iterable): probabilities.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        Dict: Dictionary mapping class label to probability.
    """
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = float(item)
    return d


class TorchPredictor:
    def __init__(self,preprocessor,model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self,batch):
        results = self.model.predict(collate_fn(batch))
        return {"output": results}

    def predict_proba(self,batch):
        results = self.model.predict_proba(collate_fn(batch))
        return {"output": results}

    def get_preprocessor(self):
        return self.preprocessor


    @classmethod
    def from_checkpoint(cls,checkpoint):
        metadata = checkpoint.get_metadata()
        preprocessor = CustomPreprocessor(class_to_index=metadata["class_to_index"])
        ckpt_path = Path(checkpoint.path)
        model = FinetunedLLM.load(ckpt_path / "args.json", ckpt_path / "model.pt")
        return cls(preprocessor= preprocessor,model=model)



def predict_proba(
    ds: ray.data.dataset.Dataset,
    predictor: TorchPredictor
) -> List:

    """
    Predict tags (with probabilites) for input data from a dataframe

    Args:
        df(pd.dataframe): dataframe with input features
        predictor(TorchPredictor): loaded predictor from a checkpoint

    Returns:
        List: list of predict labels
    """

    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    outputs = preprocessed_ds.map_batches(predictor.predict_proba)
    y_prob = np.array([d["output"] for d in outputs.take_all()])
    results = []
    for i, prob in enumerate(y_prob):
        tag = preprocessor.index_to_class[prob.argmax()]
        results.append({"prediction": tag, "probabilities": format_prob(prob, preprocessor.index_to_class)})
    return results


@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "") -> str:
    """Get the best run_id from an MLFLOW experiment

    Args:
        experiment_name (str): name of the experiment.
        metric (str): metric to filter by.
        mode (str): direction of metric (ASC/DESC).

    Returns:
        str: best run id from experiment.
    """

    sorted_runs = mlflow.search_runs(
        experiment_names = [experiment_name],
        order_by = [f"metrics.{metric} {mode}"]
    )

    run_id = sorted_runs.iloc[0].run_id
    print(run_id)
    return run_id


def get_best_checkpoint(run_id: str):
    """
    Get the best checkpoint from a specific run..

    Args:
        run_id (str): ID of the run to get the best checkpoint from.

    Returns:
        TorchCheckpoint: Best checkpoint from the run.
    """

    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    parsed = urlparse(artifact_uri)

    if parsed.scheme == "file":
        artifact_dir = Path(os.path.normpath(parsed.path.lstrip("/")))

    else:
        artifact_dir = Path(parsed.netloc) / parsed.path

    artifact_dir = artifact_dir.resolve()

    if not artifact_dir.exists():
        raise RuntimeError(f"MLflow artifact directory not found: {artifact_dir}")

    results =  Result.from_path(str(artifact_dir))
    return results.best_checkpoints[0][0]




@app.command()
def predict(
    run_id: str = typer.Option(...,help = "id of specific run to load from"),
    title: str = typer.Option(..., help = "project title"),
    description: str = typer.Option(..., help="project description")
) -> Dict:
    """Predict the tag for a project given it's title and description.

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        title (str, optional): project title. Defaults to "".
        description (str, optional): project description. Defaults to "".

    Returns:
        Dict: prediction results for the input data.
    """

    ## load components
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    ## Predict
    sample_ds = ray.data.from_items([{"title": title, "description": description, "tag": "other"}])
    results = predict_proba(ds=sample_ds,predictor=predictor)
    logger.info(json.dumps(results,cls=NumpyEncoder,indent=2))
    return results

if __name__ == "__main__":
    app()
