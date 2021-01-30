import hashlib
import os
import pandas as pd
import time

from datetime import date
from lib.models import TextSummarizationModel
from lib.predictions import predict_model
from lib.utils import root_directory
from rouge_score import rouge_scorer
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

CACHE_BASE_PATH = './data/scores'
tqdm.pandas()


def score_model(
        dataset: str, model: TextSummarizationModel, limit: Optional[int] = None,
        rouge_n: int = 3, cache_base_path: str = CACHE_BASE_PATH) -> pd.DataFrame:
    """
    Scores a dataset against a summarization model.

    :param dataset: The key of the dataset to use for the scoring
    :param model: The model which should be scored
    :param limit: Optional count of rows which should be analyzed from dataset
    :param rouge_n The longest n-gram for which Rouge should be calculated
    :param cache_base_path The base directory for the cache
    :return: A data frame containing the `text`, `summary`, `summary_predicted` and the related scores.
    """

    rouge_variants = list(map(lambda n: f"rouge{n}", list(range(1, rouge_n + 1)) + ['L']))
    scorer = rouge_scorer.RougeScorer(rouge_variants)

    def score_row(row) -> dict:
        row_scores = scorer.score(row['summary'], row['summary_predicted'])

        for r in rouge_variants:
            prefix = r.replace('rouge', 'r')
            row[f"{prefix}p"] = row_scores[r].precision
            row[f"{prefix}r"] = row_scores[r].recall
            row[f"{prefix}f"] = row_scores[r].fmeasure

        return row

    root = root_directory()
    model_key = model.to_string()
    model_key_hashed = hashlib.md5(model_key.encode()).hexdigest()
    target_directory = f"{root}/{cache_base_path}/{dataset}"
    cached_file_path = f"{target_directory}/{model_key_hashed}.scores"
    cached_file_info_path = f"{target_directory}/{model_key_hashed}.info"
    Path(target_directory).mkdir(parents=True, exist_ok=True)

    scores = None
    predictions = predict_model(dataset, model, limit)

    if os.path.isfile(cached_file_path):
        scores = pd.read_csv(cached_file_path).drop(['text', 'summary', 'summary_predicted'], axis=1)

    if scores is None or len(scores.index) < len(predictions.index):
        start = time.time()
        print(f"Scoring {len(predictions.index)} predictions of dataset `{dataset}` with model `{model.get_label()}` ...")
        scores = predictions.progress_apply(score_row, axis=1)
        end = time.time()

        # Cache predictions
        scores.to_csv(cached_file_path, index=False, header=True)
        with open(cached_file_info_path, 'w') as info_file:
            info = f"Key: {model_key}\n" \
                   f"Scores: {len(predictions.index)}\n" \
                   f"Time Elapsed (seconds): {end - start}\n" \
                   f"Executed: {date.today().strftime('%d.%m.%Y')}"
            info_file.write(info)

    return scores


def score_models(
        dataset: str, models: List[TextSummarizationModel],
        limit: Optional[int] = None, rouge_n: int = 3) -> pd.DataFrame:
    """
    Scores a dataset against multiple summarization models.

    :param dataset: The key of the dataset to use for the scoring
    :param models: The models which should be scored
    :param limit: Optional count of rows which should be analyzed from dataset
    :param rouge_n: The longest n-gram for which Rouge should be calculated
    :return:
    """

    return pd.DataFrame(
        data=[
            score_model(dataset, model, limit, rouge_n).mean()
            for model in models
        ],
        index=[model.get_label() for model in models])
