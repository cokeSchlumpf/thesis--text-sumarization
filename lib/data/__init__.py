import pandas as pd

from lib.utils import root_directory

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


def load_datasets() -> dict:
    """
    Returns a dict of known datasets (key -> label).

    :return:
    """

    return {
        'amzn': 'Amazon Reviews',
        'cnn_dailymail': 'CNN DailyMail',
        'swisstext': 'SwissText'
    }