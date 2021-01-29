from transformers import pipeline

from lib.models import TextSummarizationModel


class BartSummarizationModel(TextSummarizationModel):

    def __init__(self, min_length: int, max_length: int, do_sample: bool):
        self.summarizer = pipeline("summarization")
        self.min_length = min_length
        self.max_length = max_length
        self.do_sample = do_sample

    def to_string(self) -> str:
        return f"BartSummarizationModel({self.min_length}, {self.max_length}, {self.do_sample})"

    @staticmethod
    def get_pretrained(
            min_length: int = 3, max_length: int = 30, do_sample: bool = False) -> 'BartSummarizationModel':

        return BartSummarizationModel(min_length, max_length, do_sample)

    def predict(self, text: str) -> str:
        try:
            return self.summarizer(
                text[:1024], max_length=self.max_length,
                min_length=self.min_length, do_sample=self.do_sample)[0]['summary_text']
        except IndexError:
            return '<NO SUMMARY>'
