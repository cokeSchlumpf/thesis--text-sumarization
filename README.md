# Text Summarization Experiments

This repository contains various building blocks for executing Text Summarization experiments of my Master thesis. Feel free to reuse the stuff.

## Dashboards

The root directory contains two [Streamlit](https://www.streamlit.io/) based Dashboards. One for analysing the datasets, the other one for model scoring and comparison.

To run the dashboards:

```bash
$ TOKENIZERS_PARALLELISM=false streamlit run streamlits/test_summarizers.py
``` 