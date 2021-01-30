import hashlib
import os
import pandas as pd
import time

from datetime import date
from lib.data import load_validation_as_df
from lib.models import TextSummarizationModel
from lib.utils import root_directory
from pathlib import Path
from typing import List, Optional

CACHE_BASE_PATH = './data/predictions'


def predict_model(
        key: str, model: TextSummarizationModel,
        limit: Optional[int] = None, cache_base_path: str = CACHE_BASE_PATH) -> pd.DataFrame:
    """
    Returns predictions for a dataset based on the validation set.

    :param key: The dataset key
    :param model: The model to be used to make predictions
    :param limit: The number of maximal predicted records
    :param cache_base_path The base directory for the cache
    :return:
    """

    root = root_directory()
    model_key = model.to_string()
    model_key_hashed = hashlib.md5(model_key.encode()).hexdigest()
    target_directory = f"{root}/{cache_base_path}/{key}"
    cached_file_path = f"{target_directory}/{model_key_hashed}.predictions"
    cached_file_info_path = f"{target_directory}/{model_key_hashed}.info"
    Path(target_directory).mkdir(parents=True, exist_ok=True)

    df = load_validation_as_df(key)[:limit]
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
            info = f"Key: {model_key}\n" \
                   f"Predictions: {len(predictions.index)}\n" \
                   f"Time Elapsed (seconds): {end - start}\n" \
                   f"Executed: {date.today().strftime('%d.%m.%Y')}"
            info_file.write(info)

    # Join predictions to text and reference summary
    df = df.join(predictions[:len(df.index)])

    return df


def predict_models(dataset: str, models: List[TextSummarizationModel], limit: int = 5) -> pd.DataFrame:
    """
    Predicts samples of a dataset for multiple models.

    :param dataset: The dataset to use for the predictions
    :param models: The list of models which should predict samples
    :param limit: The number of records to be predicted
    :return:
    """

    df = load_validation_as_df(dataset)[:limit]

    for model in models:
        prediction = predict_model(dataset, model, limit)
        df[model.get_id()] = prediction['summary_predicted']

    return df
