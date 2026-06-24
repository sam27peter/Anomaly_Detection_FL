import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(

    page_title="FL Research Dashboard",

    layout="wide"

)

st.title(
    "Federated Learning Research Dashboard"
)

# ==================================================
# LOAD EXPERIMENTS
# ==================================================

experiments_dir = (

    Path("results")
    / "experiments"

)

experiments = sorted(

    [

        exp.name

        for exp in experiments_dir.glob("*")

        if exp.is_dir()

    ],

    reverse=True

)

if len(experiments) == 0:

    st.warning(
        "No experiments found."
    )

    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "Experiment Selection"
)

selected_exp = st.sidebar.selectbox(

    "Choose Experiment",

    experiments

)

exp_path = (

    experiments_dir
    / selected_exp

)

# ==================================================
# LOAD FILES
# ==================================================

metrics_file = (
    exp_path
    / "metrics.json"
)

history_file = (
    exp_path
    / "history.json"
)

config_file = (
    exp_path
    / "experiment_config.json"
)

if not metrics_file.exists():

    st.error(
        "Metrics file missing."
    )

    st.stop()

with open(metrics_file) as f:

    metrics = json.load(f)

with open(config_file) as f:

    config = json.load(f)

history = {}

if history_file.exists():

    with open(history_file) as f:

        history = json.load(f)

# ==================================================
# EXPERIMENT INFO
# ==================================================

st.header(
    "Experiment Information"
)

info_df = pd.DataFrame(

    {

        "Property": [

            "Algorithm",
            "Dataset",
            "Partition",
            "Rounds",
            "Clients"

        ],

        "Value": [

            config.get(
                "algorithm"
            ),

            config.get(
                "dataset"
            ),

            config.get(
                "partition"
            ),

            config.get(
                "rounds"
            ),

            config.get(
                "num_clients"
            )

        ]

    }

)

st.table(info_df)

# ==================================================
# METRICS
# ==================================================

st.header(
    "Performance Metrics"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{metrics.get('accuracy',0):.4f}"
)

col2.metric(
    "Precision",
    f"{metrics.get('precision',0):.4f}"
)

col3.metric(
    "Recall",
    f"{metrics.get('recall',0):.4f}"
)

col4.metric(
    "F1 Score",
    f"{metrics.get('f1_score',0):.4f}"
)

col5, col6, col7 = st.columns(3)

col5.metric(
    "ROC AUC",
    f"{metrics.get('roc_auc',0):.4f}"
)

col6.metric(
    "PR AUC",
    f"{metrics.get('pr_auc',0):.4f}"
)

col7.metric(
    "Training Time (s)",
    f"{metrics.get('training_time_sec',0):.2f}"
)

# ==================================================
# TRAINING CURVES
# ==================================================

if history:

    st.header(
        "Training Curves"
    )

    if "accuracy" in history:

        st.subheader(
            "Global Accuracy"
        )

        st.line_chart(
            history["accuracy"]
        )

    if "loss" in history:

        st.subheader(
            "Global Loss"
        )

        st.line_chart(
            history["loss"]
        )

# ==================================================
# CLIENT ACCURACY
# ==================================================

if (

    "client_accuracy"
    in history

):

    st.header(
        "Client Accuracy Distribution"
    )

    client_df = pd.DataFrame(

        {

            "Client":

                [

                    f"Client {i+1}"

                    for i in range(

                        len(

                            history[
                                "client_accuracy"
                            ]

                        )

                    )

                ],

            "Accuracy":

                history[
                    "client_accuracy"
                ]

        }

    )

    st.bar_chart(

        client_df.set_index(
            "Client"
        )

    )

# ==================================================
# FAIRNESS
# ==================================================

st.header(
    "Fairness Metrics"
)

fairness = pd.DataFrame(

    {

        "Metric": [

            "Mean Accuracy",
            "Std Accuracy",
            "Best Client",
            "Worst Client",
            "Accuracy Gap"

        ],

        "Value": [

            metrics.get(
                "mean_accuracy"
            ),

            metrics.get(
                "std_accuracy"
            ),

            metrics.get(
                "best_client_accuracy"
            ),

            metrics.get(
                "worst_client_accuracy"
            ),

            metrics.get(
                "accuracy_gap"
            )

        ]

    }

)

st.table(fairness)

# ==================================================
# COMPARISON TABLE
# ==================================================

comparison_csv = (

    Path("results")
    / "comparison"
    / "experiment_summary.csv"

)

if comparison_csv.exists():

    st.header(
        "All Experiments"
    )

    df = pd.read_csv(
        comparison_csv
    )

    st.dataframe(
        df,
        use_container_width=True
    )