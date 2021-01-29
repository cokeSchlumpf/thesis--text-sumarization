import pandas as pd

from lib.data import load_test_as_df
from lib.models import TextSummarizationModel
from pydantic import BaseModel
from rouge_score import rouge_scorer
from typing import List, Optional


class Prediction(BaseModel):
    text: str
    reference_summary: str
    predicted_summary: str


class Scores(BaseModel):
    s: str


def predict_samples(dataset: str, model: TextSummarizationModel, limit: int = 3) -> List[Prediction]:
    """
    Uses the model to make a few sample predictions from the given dataset.

    :param dataset: The key of the dataset to be used
    :param model: The model to be used
    :param limit: The number of examples
    :return: A list of Prediction's
    """
    test = load_test_as_df(dataset)[:limit]

    prediction = model.predict_all(test, inplace=True)
    return list([
        Prediction(text=row.text, reference_summary=row.summary, predicted_summary=row.summary_predicted)
        for row in prediction.itertuples()])


def score_dataset(dataset: str, model: TextSummarizationModel, limit: Optional[int] = None, rouge_n: int = 3) -> pd.DataFrame:
    """
    Scores a dataset against a summarization model.

    :param dataset: The key of the dataset to use for the scoring
    :param model: The model which should be scored
    :param limit: Optional count of rows which should be analyzed from dataset
    :return: A data frame containing the `text`, `summary`, `summary_predicted` and the related scores.
    """

    rouge_variants = list(map(lambda n: f"rouge{n}", list(range(1, rouge_n + 1)) + ['L']))
    scorer = rouge_scorer.RougeScorer(rouge_variants)

    def score_row(row) -> dict:
        scores = scorer.score(row['summary'], row['summary_predicted'])

        for r in rouge_variants:
            prefix = r.replace('rouge', 'r')
            row[f"{prefix}p"] = scores[r].precision
            row[f"{prefix}r"] = scores[r].recall
            row[f"{prefix}f"] = scores[r].fmeasure

        return row

    test = load_test_as_df(dataset)

    if limit is not None:
        test = test[:limit]

    prediction = model.predict_all(test, inplace=True)
    prediction = prediction.apply(score_row, axis=1)
    return prediction


def score_dataset_summary(dataset: str, model: TextSummarizationModel, limit: Optional[int] = None) -> Scores:
    """
    Score a dataset against a model.

    :param dataset: The key for the dataset
    :param model:  The model which should be scored
    :param limit: Optional count of rows which should be analyzed from dataset
    :return:
    """

    return Scores()

