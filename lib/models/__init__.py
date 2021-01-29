import pandas as pd

from abc import abstractmethod


class TextSummarizationModel:
    """
    An interface for text summarization models.
    """

    @abstractmethod
    def predict(self, text: str) -> str:
        """
        Return the summary of a single text.

        :param text: The text to summarize.
        :return: The summary of the text
        """
        pass

    def predict_all(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Predict the summary of a set of texts. The data frame should provide a column `text` including the text to
        summarize.

        :param df: The data frame with a column `text`
        :param inplace: Whether the data frame should be extended with the predictions or not
        :return: A data frame, the predictions will be in column `summary_predicted`
        """
        if inplace:
            df['summary_predicted'] = df['text'].pipe(lambda seq: seq.apply(self.predict))
            return df
        else:
            result = pd.DataFrame(data={'summary_predicted': df['text'].pipe(lambda seq: seq.apply(self.predict))})
            return result

    @abstractmethod
    def to_string(self) -> str:
        """
        This method should return a serialization key for the class including name and properties.

        :return: The string unique for the class instance.
        """
        pass
