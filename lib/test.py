from lib.data import load_training_as_df

def main():
    df = load_training_as_df('amzn')
    print(df.head())


if __name__ == '__main__':
    main()
