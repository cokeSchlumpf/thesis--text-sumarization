import streamlit as st

from lib.data import get_datasets
from lib.models import get_models, hash_functions
from lib.predictions import predict_models
from lib.scores import score_models
from lib.utils import strip_margin


@st.cache(persist=False, hash_funcs=hash_functions())
def initialize():
    return get_models(), get_datasets()


models, datasets = initialize()


#
# Sidebar
#
st.sidebar.write("# Configure View")

st.sidebar.write('**Dataset and model**')
dataset_title = st.sidebar.selectbox(
    'Select the dataset', list(datasets.values()))

st.sidebar.write("Select models to compare")
models_selection = [st.sidebar.checkbox(models[key]['model'].get_label(), models[key]['default_selected'], key) for key in models]

st.sidebar.write("**Scoring Configuration**")
rouge_n = st.sidebar.slider('n-grams', min_value=1, max_value=4, value=2)


#
# Data
#
dataset_key = list(datasets.keys())[list(datasets.values()).index(dataset_title)]
models_selected = list(map(lambda m: models[m[1]]['model'], filter(lambda m: m[0] is True, zip(models_selection, models))))
models_scores = score_models(dataset_key, models_selected)
models_samples = predict_models(dataset_key, models_selected, 10)

#
# Output
#
st.write("# Text Summarization Model Summary")

st.dataframe(models_scores)

with st.beta_expander("See explanation"):
    st.write(strip_margin("""|
        | * ROUGE-n recall=40% means that 40% of the n-grams in the reference summary are also present in the generated summary.
        | * ROUGE-n precision=40% means that 40% of the n-grams in the generated summary are also present in the reference summary.
        | * ROUGE-n F1-score=40% is more difficult to interpret, like any F1-score."""))

st.write("# Samples")
for row in models_samples.iterrows():
    index = row[0] + 1
    sample = row[1]

    st.write(f"## #{index}")

    with st.beta_expander("Sample Text"):
        st.write(sample['text'])

    with st.beta_expander("Reference Summary", True):
        st.write(sample['summary'])

    for model in models_selected:
        with st.beta_expander(model.get_label(), True):
            st.write(sample[model.get_id()])
