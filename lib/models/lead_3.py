import spacy

from lib.models import TextSummarizationModel

LANGUAGE_MODEL_EN = 'en_core_web_sm'
LANGUAGE_MODEL_DE = 'de_core_news_sm'


class Lead3SummarizationModel(TextSummarizationModel):

    def __init__(self, language_model: str, label: str):
        self.language_model = language_model
        self.nlp = None
        self.label = label

    @staticmethod
    def get_pretrained(language_model: str = LANGUAGE_MODEL_EN, label: str = 'LEAD-3') -> TextSummarizationModel:
        return Lead3SummarizationModel(language_model, label)

    def get_label(self) -> str:
        return self.label

    def to_string(self) -> str:
        return f"Lead3SummarizationModel({self.language_model})"

    def predict(self, text: str) -> str:
        if self.nlp is None:
            self.nlp = spacy.load(self.language_model)

        doc = self.nlp(text)
        return str.join(' ', map(lambda s: str(s).strip(), list(doc.sents)[:3]))
