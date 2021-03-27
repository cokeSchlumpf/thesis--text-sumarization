import pandas as pd

from abc import abstractmethod
from tqdm import tqdm

tqdm.pandas()


class TextSummarizationModel:
    """
    An interface for text summarization models.
    """

    @abstractmethod
    def get_label(self) -> str:
        """
        Should return a unique label for the model.

        :return:
        """
        pass

    def get_id(self) -> str:
        """
        Returns a unique label which can be used as technical name.

        :return:
        """
        return str.join('', [c for c in self.get_label() if c.isalnum()]).lower()

    @abstractmethod
    def predict(self, text: str) -> str:
        """
        Return the summary of a single text.

        :param text: The text to summarize.
        :return: The summary of the text
        """
        pass

    def predict_all(
            self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:

        """
        Predict the summary of a set of texts. The data frame should provide a column `text` including the text to
        summarize.

        :param df: The data frame with a column `text`
        :param inplace: Whether the data frame should be extended with the predictions or not
        :return: A data frame, the predictions will be in column `summary_predicted`
        """

        print(f"Creating {len(df.index)} predictions with model `{self.to_string()}` ...")

        if inplace:
            df['summary_predicted'] = df['text'].progress_apply(self.predict)
            result = df
        else:
            result = pd.DataFrame(data={'summary_predicted': df['text'].progress_apply(self.predict)})
            result = result

        return result

    @abstractmethod
    def to_string(self) -> str:
        """
        This method should return a serialization key for the class including name and properties.

        :return: The string unique for the class instance.
        """
        pass

    def dict(self) -> dict:
        """
        Similar to Pydantics dict method this method returns a representation of the models properties as dict.
        Returns: The dict representation of the model
        """

        return {
            'id': self.get_id(),
            'label': self.get_label()
        }


def get_models():
    """
    Loads all known models and returns a dictionary with available models.

    :return:
    """
    from .lead_3 import Lead3SummarizationModel, LANGUAGE_MODEL_DE, LANGUAGE_MODEL_EN
    from .bart import BartSummarizationModel
    from .t5 import T5SummarizationModel

    _lead3_en = Lead3SummarizationModel.get_pretrained(LANGUAGE_MODEL_EN, 'lead-3 (EN)')
    _lead3_de = Lead3SummarizationModel.get_pretrained(LANGUAGE_MODEL_DE, 'lead-3 (DE)')
    _bart = BartSummarizationModel.get_pretrained()
    _t5 = T5SummarizationModel.get_pretrained()

    return [_lead3_de, _lead3_en, _bart, _t5]


def get_model_by_name(name: str) -> TextSummarizationModel:
    models = get_models()
    models = list(filter(lambda m: m.get_id() == name, models))

    if len(models) == 0:
        raise Exception(f"No model found with name `{name}`.")

    return models[0]


def hash_functions():
    """
    Set of hash functions for all model types.

    :return:
    """
    from .lead_3 import Lead3SummarizationModel, LANGUAGE_MODEL_DE, LANGUAGE_MODEL_EN
    from .bart import BartSummarizationModel
    from .t5 import T5SummarizationModel

    def hash_model(model):
        return model.to_string()

    return {
        Lead3SummarizationModel: hash_model,
        BartSummarizationModel: hash_model,
        T5SummarizationModel: hash_model
    }
