import click

from lib.data import get_datasets, get_dataset_by_name
from lib.models import get_models, get_model_by_name
from lib.predictions import predict_model
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


@click.group()
def main():
    pass


main.add_command(list_models)
main.add_command(list_datasets)
main.add_command(predict)

if __name__ == '__main__':
    main()
