import glob
import hashlib
import os
import pandas as pd
import time

from datetime import datetime, timedelta
from lib.data import Dataset
from lib.models import TextSummarizationModel
from lib.predictions import predict_model
from lib.utils import root_directory
from rouge_score import rouge_scorer
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm
from typing import List, Optional

CACHE_BASE_PATH = './data/scores'
tqdm.pandas()


class ScoreInfo(BaseModel):

    model: str
    dataset: str
    count: int
    started: float
    finished: float
    elapsed: str
    scores: dict

    def get_started(self) -> str:
        return datetime.fromtimestamp(self.started).strftime("%Y-%m-%d, %H:%M:%S")

    @staticmethod
    def get_table_header(rouge_n: int = 3) -> List[str]:
        rouge_variants = list(map(lambda n: f"r{n}", list(range(1, rouge_n + 1)) + ['L']))

        header = [
            "MODEL",
            "DATASET",
            "EXECUTED",
            "DURATION",
            "COUNT"
        ]

        for r in rouge_variants:
            header = header + [f"{r}p", f"{r}r", f"{r}f"]

        return header

    def get_table_row(self, rouge_n: int = 3) -> List[str]:
        rouge_variants = list(map(lambda n: f"r{n}", list(range(1, rouge_n + 1)) + ['L']))

        row = [
            self.model,
            self.dataset,
            self.get_started(),
            self.elapsed,
            self.count
        ]

        for r in rouge_variants:
            row = row + [
                self.scores.get(f"{r}p", '-'),
                self.scores.get(f"{r}r", '-'),
                self.scores.get(f"{r}f", '-')
            ]

        return row


def list_scores(cache_base_path: str = CACHE_BASE_PATH) -> List[ScoreInfo]:
    """
    Reads the cached scores and returns them.

    Args:
        cache_base_path: he base directory for the cache

    Returns:
        The cached scores
    """
    root = root_directory()
    target_directory = f"{root}/{cache_base_path}"
    return [ScoreInfo.parse_file(f) for f in glob.iglob(target_directory + '/**/*.info', recursive=True)]


def score_model(
        dataset: Dataset, model: TextSummarizationModel, limit: Optional[int] = None,
        rouge_n: int = 3, cache_base_path: str = CACHE_BASE_PATH) -> pd.DataFrame:
    """
    Scores a dataset against a summarization model.

    :param dataset: The dataset for which the core should be calculated
    :param model: The model which should be scored
    :param limit: Optional count of rows which should be analyzed from dataset
    :param rouge_n The longest n-gram for which Rouge should be calculated
    :param cache_base_path The base directory for the cache
    :return: A data frame containing the `text`, `summary`, `summary_predicted` and the related scores.
    """

    dataset_key = dataset.id
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
    target_directory = f"{root}/{cache_base_path}/{dataset_key}"
    cached_file_path = f"{target_directory}/{model_key_hashed}.scores"
    cached_file_info_path = f"{target_directory}/{model_key_hashed}.info"
    Path(target_directory).mkdir(parents=True, exist_ok=True)

    scores = None
    predictions = predict_model(dataset, model, limit)

    if os.path.isfile(cached_file_path):
        scores = pd.read_csv(cached_file_path).drop(['text', 'summary', 'summary_predicted'], axis=1)

    if scores is None or len(scores.index) < len(predictions.index):
        start = time.time()
        print(f"Scoring {len(predictions.index)} predictions of dataset `{dataset_key}` with model `{model.get_label()}` ...")
        scores = predictions.progress_apply(score_row, axis=1)
        end = time.time()

        # Cache predictions
        scores.to_csv(cached_file_path, index=False, header=True)
        with open(cached_file_info_path, 'w') as info_file:
            scores_summary = {}

            for r in rouge_variants:
                prefix = r.replace('rouge', 'r')
                scores_summary[f"{prefix}p"] = round(scores[f"{prefix}p"].mean(), 4)
                scores_summary[f"{prefix}r"] = round(scores[f"{prefix}r"].mean(), 4)
                scores_summary[f"{prefix}f"] = round(scores[f"{prefix}f"].mean(), 4)

            info = ScoreInfo(
                model=model.get_id(), dataset=dataset.id, count=scores.shape[0],
                started=start, finished=end, elapsed=str(timedelta(seconds=end - start)),
                scores=scores_summary)
            info = info.json()
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
