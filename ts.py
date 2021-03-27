import click

from lib.data import get_datasets, get_dataset_by_name
from lib.models import get_models, get_model_by_name
from lib.predictions import predict_model, list_predictions
from lib.scores import score_model, list_scores, ScoreInfo
from tabulate import tabulate
from typing import Optional


@click.command(name='models')
def list_models():
    """
    Lists known models
    """
    models = [m.dict().values() for m in get_models()]
    print(tabulate(models, headers=['NAME', 'LABEL'], tablefmt='plain'))


@click.command(name='datasets')
def list_datasets():
    """
    List known datasets
    """
    datasets = [ds.dict(include={'id', 'language', 'name'}).values() for ds in get_datasets()]
    print(tabulate(datasets, headers=['NAME', 'LABEL', 'LANGUAGE'], tablefmt='plain'))


@click.argument('dataset')
@click.argument('model')
@click.option('-l', '--limit', 'limit', type=int)
@click.command(name='predict')
def predict(dataset: str, model: str, limit: Optional[int] = None):
    """
    Creates summaries with the given model for the given dataset
    """
    mdl = get_model_by_name(model)
    ds = get_dataset_by_name(dataset)
    print(f"Running prediction for dataset `{ds.name}` with model `{mdl.get_label()}`")
    predict_model(ds, mdl, limit)


@click.command(name='predictions')
def predictions():
    """
    Lists summaries for existing predictions.
    """
    result = [p.get_table_row() for p in list_predictions()]
    print(tabulate(result, headers=['MODEL', 'DATASET', 'EXECUTED', 'DURATION', 'COUNT'], tablefmt='plain'))


@click.option('-n', '--rouge-n', 'rouge_n', type=int, default=2)
@click.command(name='scores')
def scores(rouge_n: int = 2):
    """
    Lists existing scores
    """
    result = [s.get_table_row(rouge_n=rouge_n) for s in list_scores()]
    print(tabulate(result, headers=ScoreInfo.get_table_header(rouge_n=rouge_n), tablefmt='plain'))


@click.argument('dataset')
@click.argument('model')
@click.option('-l', '--limit', 'limit', type=int)
@click.option('-n', '--rouge-n', 'rouge_n', type=int, default=2)
@click.command(name='score')
def score(dataset: str, model: str, rouge_n: int = 2, limit: Optional[int] = None):
    """
    Calculates the Rouge-N score for predictions of a model for a dataset
    """
    mdl = get_model_by_name(model)
    ds = get_dataset_by_name(dataset)
    print(f"Running score calculation for dataset `{ds.name}` with model `{mdl.get_label()}`")
    score_model(ds, mdl, limit, rouge_n)


@click.group()
def main():
    pass


main.add_command(list_models)
main.add_command(list_datasets)
main.add_command(predict)
main.add_command(predictions)
main.add_command(score)
main.add_command(scores)

if __name__ == '__main__':
    main()
