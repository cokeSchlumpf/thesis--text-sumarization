import glob
import hashlib
import os
import pandas as pd
import time

from datetime import datetime, timedelta
from lib.data import Dataset, load_validation_as_df
from lib.models import TextSummarizationModel
from lib.utils import root_directory
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

CACHE_BASE_PATH = './data/predictions'


class PredictionInfo(BaseModel):

    model: str
    dataset: str
    count: int
    started: float
    finished: float
    elapsed: str

    def get_started(self) -> str:
        return datetime.fromtimestamp(self.started).strftime("%Y-%m-%d, %H:%M:%S")

    def get_table_row(self) -> List[str]:
        return [
            self.model,
            self.dataset,
            self.get_started(),
            self.elapsed,
            self.count
        ]


def list_predictions(cache_base_path: str = CACHE_BASE_PATH) -> List[PredictionInfo]:
    """
    Reads the cached predictions and returns the info for all predictions.

    Args:
        cache_base_path: The base directory for the cache

    Returns:
        The summaries.
    """
    root = root_directory()
    target_directory = f"{root}/{cache_base_path}"
    return [PredictionInfo.parse_file(f) for f in glob.iglob(target_directory + '/**/*.info', recursive=True)]


def predict_model(
        dataset: Dataset, model: TextSummarizationModel,
        limit: Optional[int] = None, cache_base_path: str = CACHE_BASE_PATH) -> pd.DataFrame:
    """
    Returns predictions for a dataset based on the validation set.

    :param dataset: The dataset to be used
    :param model: The model to be used to make predictions
    :param limit: The number of maximal predicted records
    :param cache_base_path The base directory for the cache
    :return:
    """
    key = dataset.id
    root = root_directory()
    model_key = model.to_string()
    model_key_hashed = hashlib.md5(model_key.encode()).hexdigest()
    target_directory = f"{root}/{cache_base_path}/{key}"
    cached_file_path = f"{target_directory}/{model_key_hashed}.predictions"
    cached_file_info_path = f"{target_directory}/{model_key_hashed}.info"
    Path(target_directory).mkdir(parents=True, exist_ok=True)

    df = load_validation_as_df(key)

    if limit is not None:
        df = df[:limit]

    predictions = None

    # Load cached predictions if available
    if os.path.isfile(cached_file_path):
        predictions = pd.read_csv(cached_file_path)

    # Execute prediction of necessary (no cache available)
    if predictions is None or len(predictions.index) < len(df.index):
        start = time.time()
        predictions = model.predict_all(df)
        end = time.time()

        # Cache predictions
        predictions.to_csv(cached_file_path, index=False, header=True)
        with open(cached_file_info_path, 'w') as info_file:
            info = PredictionInfo(
                model=model.get_id(), dataset=dataset.id, count=predictions.shape[0],
                started=start, finished=end, elapsed=str(timedelta(seconds=end - start)))
            info = info.json()
            info_file.write(info)

    # Join predictions to text and reference summary
    df = df.join(predictions[:len(df.index)])

    return df


def predict_models(dataset: Dataset, models: List[TextSummarizationModel], limit: int = 5) -> pd.DataFrame:
    """
    Predicts samples of a dataset for multiple models.

    :param dataset: The dataset to use for the predictions
    :param models: The list of models which should predict samples
    :param limit: The number of records to be predicted
    :return:
    """

    df = load_validation_as_df(dataset.id)[:limit]

    for model in models:
        prediction = predict_model(dataset, model, limit)
        df[model.get_id()] = prediction['summary_predicted']

    return df
