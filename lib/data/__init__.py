import pandas as pd

from .dataset import Dataset
from lib.utils import root_directory
from typing import List

SOURCE_BASE_PATH = './data/prepared'


def _load_target_and_source(
        key: str, source_base_path: str, dataset: str) -> pd.DataFrame:
    root = root_directory()
    df_source = pd.read_csv(f"{root}/{source_base_path}/{key}/{dataset}.source")
    df_target = pd.read_csv(f"{root}/{source_base_path}/{key}/{dataset}.target")

    return df_source.join(df_target)


def load_training_as_df(
        key: str, source_base_path: str = SOURCE_BASE_PATH, include_validation: bool = False) -> pd.DataFrame:
    """
    Loads the training data for a dataset.

    :param key: The key of the dataset
    :param source_base_path: The base path for the prepared data, relative to project root
    :param include_validation: Whether validation data should be included in the set
    :return:
    """

    train = _load_target_and_source(key, source_base_path, 'train')

    if include_validation:
        validation = _load_target_and_source(key, source_base_path, 'validation')
        return pd.concat([train, validation])
    else:
        return train


def load_test_as_df(
        key: str, source_base_path: str = SOURCE_BASE_PATH) -> pd.DataFrame:
    """
    Loads the test data for a dataset.

    :param key: The key of the dataset
    :param source_base_path: The base path for the prepared data, relative to project root
    :return:
    """

    return _load_target_and_source(key, source_base_path, 'test')


def load_validation_as_df(
        key: str, source_base_path: str = SOURCE_BASE_PATH) -> pd.DataFrame:
    """
    Loads the validation data for a dataset.

    :param key: The key of the dataset
    :param source_base_path:  The base path for the prepared data, relative to project root
    :return:
    """

    return _load_target_and_source(key, source_base_path, 'validation')


def get_datasets() -> List[Dataset]:
    """
    Returns a dict of known datasets (key -> label).

    :return:
    """

    amzn = Dataset(
        id='amzn', name='Amazon Reviews', language='en',
        description="This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.")

    cnn = Dataset(
        id='cnn_dailymail', name='CNN/ DailyMail', language='en',
        description='The well-known CNN/ DailyMail data set for text summarization (version 3.0.0). The data has been fetched via HuggingFace Datasets')

    swisstext = Dataset(
        id='swisstext', name='SwissText 2019', language='de',
        description='The dataset was published for the SwissText conference 2019. ')

    return [amzn, cnn, swisstext]


def get_dataset_by_name(name: str) -> Dataset:
    datasets = get_datasets()
    datasets = list(filter(lambda d: d.id == name, datasets))

    if len(datasets) == 0:
        raise Exception(f"No dataset `{name}` found.")

    return datasets[0]
