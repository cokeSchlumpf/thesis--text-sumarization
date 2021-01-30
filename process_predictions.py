import sys

from lib.data import load_datasets
from lib.models import load_models
from lib.predictions import predict_model

datasets = load_datasets()
models = load_models()


def print_usage():
    print("Usage: process_predictions DATASET_KEY MODEL_KEY")
    print("")
    print("Available datasets:")
    for key in datasets.keys():
        print(f"- {key}: {datasets[key]}")
    print("")

    print("Available models:")
    for key in models.keys():
        print(f"- {key}: {models[key]['model'].to_string()}")


def main():
    if len(sys.argv) is not 3:
        print("process_predictions: Missing program arguments")
        print("")
        print_usage()
    else:
        dataset_key = sys.argv[1]
        model_name = sys.argv[2]

        if model_name not in models:
            print(f"process_predictions: '{model_name}' is an unknown model")
            print("")
            print_usage()
        elif dataset_key not in datasets:
            print(f"process_predictions: '{dataset_key}' is an unknown dataset")
            print("")
            print_usage()
        else:
            model = models[model_name]['model']
            predict_model(dataset_key, model)


if __name__ == "__main__":
    main()
