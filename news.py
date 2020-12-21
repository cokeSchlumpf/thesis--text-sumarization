import pandas as pd

DATA_PATH = '/Users/michaelwellner/Workspaces/thesis--news-crawler/export'
METADATA_FILE = f"{DATA_PATH}/_items.csv"

def load_text(row, data_path = DATA_PATH):
    hash = row['hash']
    with open(f"{data_path}/{hash}.content.txt", 'r') as file:
        return file.read()