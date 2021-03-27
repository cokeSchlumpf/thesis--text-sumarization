from lib.models import TextSummarizationModel


class BartSummarizationModel(TextSummarizationModel):

    def __init__(self, min_length: int, max_length: int, do_sample: bool, label: str):
        self.summarizer = None
        self.min_length = min_length
        self.max_length = max_length
        self.do_sample = do_sample
        self.label = label

    def get_label(self) -> str:
        return self.label

    def to_string(self) -> str:
        return f"BartSummarizationModel({self.min_length}, {self.max_length}, {self.do_sample})"

    @staticmethod
    def get_pretrained(
            min_length: int = 3, max_length: int = 30, do_sample: bool = False, label: str = 'Bart')\
            -> 'BartSummarizationModel':

        return BartSummarizationModel(min_length, max_length, do_sample, label)

    def predict(self, text: str) -> str:
        if self.summarizer is None:
            from transformers import pipeline
            pipeline("summarization", device=0)

        try:
            return self.summarizer(
                text[:1024], max_length=self.max_length,
                min_length=self.min_length, do_sample=self.do_sample)[0]['summary_text']
        except IndexError:
            return '<NO SUMMARY>'
