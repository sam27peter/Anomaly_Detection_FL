import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FL Dashboard",
    layout="wide"
)

st.title(
    "Federated Learning Dashboard"
)

results_dir = (
    Path("results")
)

metric_files = list(
    results_dir.rglob("*.json")
)

st.sidebar.header(
    "Available Result Files"
)

selected = st.sidebar.selectbox(

    "Select File",

    [f.name for f in metric_files]

)

selected_path = None

for file in metric_files:

    if file.name == selected:

        selected_path = file

        break

if selected_path:

    with open(selected_path) as f:

        data = json.load(f)

    st.subheader(
        selected
    )

    st.json(data)