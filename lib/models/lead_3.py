import spacy

from lib.models import TextSummarizationModel

LANGUAGE_MODEL_EN = 'en_core_web_sm'
LANGUAGE_MODEL_DE = 'de_core_news_sm'


class Lead3SummarizationModel(TextSummarizationModel):

    def __init__(self, language_model: str = LANGUAGE_MODEL_EN):
        self.language_model = language_model
        self.nlp = spacy.load(language_model)

    def to_string(self) -> str:
        return f"Lead3SummarizationModel({self.language_model})"

    def predict(self, text: str) -> str:
        doc = self.nlp(text)
        return str.join(' ', map(lambda s: str(s).strip(), list(doc.sents)[:3]))


def get_pretrained(language_model: str = LANGUAGE_MODEL_EN) -> TextSummarizationModel:
    return Lead3SummarizationModel(language_model)