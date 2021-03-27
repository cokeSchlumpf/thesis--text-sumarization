from lib.models import TextSummarizationModel


class T5SummarizationModel(TextSummarizationModel):

    def __init__(
            self, min_length: int, max_length: int, length_penalty: float,
            num_beams: int, early_stopping: bool, label: str):

        self.min_length = min_length
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.num_beans = num_beams
        self.early_stopping = early_stopping
        self.model = None
        self.tokenizer = None
        self.label = label

    @staticmethod
    def get_pretrained(
            min_length: int = 3, max_length: int = 30,
            length_penalty: float = 2.0, num_beams: int = 4,
            early_stopping: bool = True, label: str = 't5-base') -> 'T5SummarizationModel':

        return T5SummarizationModel(min_length, max_length, length_penalty, num_beams, early_stopping, label)

    def get_label(self) -> str:
        return self.label

    def to_string(self) -> str:
        return f"T5SummarizationModel(" \
               f"{self.min_length}, {self.max_length}, {self.length_penalty}, " \
               f"{self.num_beans}, {self.early_stopping})"

    def predict(self, text: str) -> str:
        if self.model is None:
            from transformers import AutoModelWithLMHead, AutoTokenizer
            self.model = AutoModelWithLMHead.from_pretrained('t5-base')
            self.tokenizer = AutoTokenizer.from_pretrained('t5-base')

        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512)
        outputs = self.model.generate(
            inputs, max_length=self.max_length, min_length=self.min_length, length_penalty=self.length_penalty,
            num_beams=self.num_beans, early_stopping=self.early_stopping)
        text = self.tokenizer.decode(outputs[0])

        return text[6:]
