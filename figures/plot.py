import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_pickle_files(directory):
    table_filename = os.path.join(directory, "table.csv")
    table = pd.read_csv(table_filename)
    model = table["model"].unique()
    assert model.size == 1
    return model[0], table["pickle"]


def parse_experiment(directory):
    model, pickle_files = get_pickle_files(directory)
    df = pd.concat(pd.DataFrame(pd.read_pickle(f)) for f in pickle_files)
    return model, df


"""
IEEE experiment
"""


def get_success_ieee_pce(df: pd.DataFrame):
    return (df["reject"] / df["tests"]) <= (1 - df["confidence"])


def get_success_ieee_bonferroni(df: pd.DataFrame):
    return df["reject"] == 0


def get_experiment_ieee_df(df: pd.DataFrame, get_success):
    df = df.copy()
    df["success"] = get_success(df)
    df = df[
        (df["reference_dataset"] == df["target_dataset"])
        & (df["reference_subject"] == df["target_subject"])
    ]
    df.reset_index(inplace=True, drop=True)

    df = df.groupby(
        [
            "reference_perturbation",
            "confidence",
            "reference_fwhm",
        ],
        group_keys=True,
    )["success"].mean()
    df = df.to_frame()
    df.reset_index(inplace=True)
    return df


def _plot_experiment_ieee(df: pd.DataFrame, get_success, test):
    df = get_experiment_ieee_df(df, get_success)

    perturbation_modes = df["reference_perturbation"].unique()

    rows = perturbation_modes.size
    cols = 1
    row_titles = list(map(str.upper, perturbation_modes.tolist()))
    column_titles = ""
    x_title = "FWHM (mm)"
    y_title = "α threshold"

    fig = make_subplots(
        rows=rows,
        cols=cols,
        column_titles=column_titles,
        row_titles=row_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title=x_title,
        y_title=y_title,
        vertical_spacing=0.001,
        horizontal_spacing=0.001,
    )

    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] in row_titles:
            annotation["textangle"] = 0

    for i, perturbation in enumerate(perturbation_modes, start=1):
        dfp = df[df["reference_perturbation"] == perturbation]
        success = dfp["success"].to_numpy()
        fwhm = dfp["reference_fwhm"].unique()
        confidence = dfp["confidence"].unique()

        success = success.reshape((confidence.size, fwhm.size))

        x = fwhm
        y = [f"{1-c:.3f}" for c in confidence]
        colormap = px.colors.make_colorscale(["black", "#009E73"])

        subfig = go.Heatmap(
            z=success,
            x=x,
            y=y,
            colorscale=colormap,
            zmin=0,
            zmax=1,
        )

        fig.add_trace(subfig, row=i, col=1)

    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"ieee_{test}",
            "scale": 1,
        }
    }

    fig.show(config=config)


def plot_experiment_ieee_pce(df: pd.DataFrame):
    print("IEEE PCE")
    _plot_experiment_ieee(df, get_success_ieee_pce, "pce")


def plot_experiment_ieee_bonferroni(df: pd.DataFrame):
    print("IEEE Bonferroni")
    _plot_experiment_ieee(df, get_success_ieee_bonferroni, "bonferroni")


def plot_experiment_ieee(df: pd.DataFrame):
    # 1- group by reference_perturbation
    # 2- group by reference_dataset, reference_subject
    # 3- split between (reference_dataset, reference_subject) == (target_dataset, target_subject)
    # 4- Compute ratio successes
    df = df.groupby(
        [
            "reference_perturbation",
            "reference_dataset",
            "reference_subject",
            "target_dataset",
            "target_subject",
            "reference_fwhm",
            "confidence",
        ],
        group_keys=True,
    ).apply(lambda x: x)
    pce = df[df["method"] == "pce"]
    bonferroni = df[df["method"] == "fwe_bonferroni"]

    plot_experiment_ieee_pce(pce)
    plot_experiment_ieee_bonferroni(bonferroni)


def plot_experiment_template_pce(df: pd.DataFrame):
    pass


def plot_experiment_template_bonferroni(df: pd.DataFrame):
    pass


def plot_experiment_template(df: pd.DataFrame):
    print(df)
    df = df.groupby(
        [
            "reference_perturbation",
            "reference_dataset",
            "reference_subject",
            "reference_template",
            "reference_fwhm",
            "target_dataset",
            "target_subject",
            "target_template",
            "confidence",
        ],
        group_keys=True,
    ).apply(lambda x: x)

    pce = df[df["method"] == "pce"]
    bonferroni = df[df["method"] == "fwe_bonferroni"]

    plot_experiment_template_pce(pce)
    plot_experiment_template_bonferroni(bonferroni)


_plot_experiment = {"ieee": plot_experiment_ieee, "template": plot_experiment_template}


def plot_experiment(model, df):
    return _plot_experiment[model](df)


def parse_args():
    parser = argparse.ArgumentParser("plot-stabilitest")
    parser.add_argument("--directory", help="Directory that contains experiment")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model, df = parse_experiment(args.directory)
    plot_experiment(model, df)


if "__main__" == __name__:
    main()
