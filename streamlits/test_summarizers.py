import sys
sys.path.append("/Users/michaelwellner/Workspaces/thesis--text-summarization")

import streamlit as st
import pandas as pd

from lib.models.bart import BartSummarizationModel
from lib.models.lead_3 import Lead3SummarizationModel, get_pretrained, LANGUAGE_MODEL_DE, LANGUAGE_MODEL_EN
from lib.models.t5 import T5SummarizationModel
from lib.data.score_model import score_dataset, predict_samples


SAMPLE_SIZE = 3


def hash_model(model):
    return model.to_string()


hash_funcs = {
    Lead3SummarizationModel: hash_model,
    BartSummarizationModel: hash_model,
    T5SummarizationModel: hash_model
}


@st.cache(persist=False, hash_funcs=hash_funcs)
def initialize():
    _models = {
        'lead_3_en': {
            'label': 'Lead 3 (EN)',
            'model': get_pretrained(LANGUAGE_MODEL_EN),
            'default_selected': False
        },
        'lead_3_de': {
            'label': 'Lead 3 (DE)',
            'model': get_pretrained(LANGUAGE_MODEL_DE),
            'default_selected': False
        },
        'bart': {
            'label': 'BART',
            'model': BartSummarizationModel.get_pretrained(),
            'default_selected': False
        },
        't5': {
            'label': 'T5',
            'model': T5SummarizationModel.get_pretrained(),
            'default_selected': True
        }
    }

    _datasets = {
        'amzn': 'Amazon Reviews',
        'cnn_dailymail': 'CNN DailyMail',
        'swisstext': 'SwissText'
    }

    return _models, _datasets


models, datasets = initialize()

#
# Sidebar
#
st.sidebar.write("# Configure View")

st.sidebar.write('**Dataset and model**')
dataset_title = st.sidebar.selectbox(
    'Select the dataset',
    (datasets['amzn'], datasets['swisstext']))

st.sidebar.write("Select models to compare")
models_selection = [st.sidebar.checkbox(models[key]['label'], models[key]['default_selected'], key) for key in models]

st.sidebar.write("**Scoring Configuration**")
rouge_n = st.sidebar.slider('n-grams', min_value=1, max_value=4, value=2)


#
# Data
#
@st.cache(persist=False, hash_funcs=hash_funcs)
def score(model, dataset):
    scores = score_dataset(dataset, model['model'], SAMPLE_SIZE, rouge_n)
    scores = scores.drop(['text', 'summary', 'summary_predicted'], axis=1).mean()
    return scores


@st.cache(persist=False, hash_funcs=hash_funcs)
def get_samples(model, dataset, count):
    return predict_samples(dataset, model)


dataset_key = list(datasets.keys())[list(datasets.values()).index(dataset_title)]
models_selected = list(map(lambda m: models[m[1]], filter(lambda m: m[0] is True, zip(models_selection, models))))
models_scores = pd.DataFrame(
    data=[score(model, dataset_key) for model in models_selected],
    index=[model['label'] for model in models_selected])

samples_count = 3
samples = []

if len(models_selected) > 0:
    model_samples = [get_samples(m['model'], dataset_key, samples_count) for m in models_selected]

    for i in range(0, samples_count):
        sample = {
            'index': i + 1,
            'text': model_samples[0][i].text,
            'reference_summary': model_samples[0][i].reference_summary
        }

        for j in range(0, len(models_selected)):
            sample[models_selected[j]['label']] = model_samples[j][i].predicted_summary

        samples = samples + [sample]


#
# Output
#
st.write("# Text Summarization Model Summary")

st.dataframe(models_scores)

desc = """
* ROUGE-n recall=40% means that 40% of the n-grams in the reference summary are also present in the generated summary.
* ROUGE-n precision=40% means that 40% of the n-grams in the generated summary are also present in the reference summary.
* ROUGE-n F1-score=40% is more difficult to interpret, like any F1-score.
""".strip()

with st.beta_expander("See explanation"):
    st.write(desc)

st.write('# Samples')

for sample in samples:
    st.write(f"## #{sample['index']}")

    with st.beta_expander("Sample Text"):
        st.write(sample['text'])

    with st.beta_expander("Reference Summary", True):
        st.write(sample['reference_summary'])

    for model in models_selected:
        with st.beta_expander(model['label'], True):
            st.write(sample[model['label']])
