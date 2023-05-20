import hashlib
import itertools
import sys
import argparse
import glob
import os
import pickle
import tqdm

import numpy as np
import polars as pd
import plotly.express as px
import plotly.io as pio
import scipy.stats
from plotly.subplots import make_subplots

import dataset_mapping as dsmap


# pio.kaleido.scope.mathjax = None

tests_name = {
    "exclude": "all-exclude",
    "include": "all-include",
    "one": "one",
    "inter": "one",
}

methods_name = [
    "pce",
    "fdr_BH",
    "fdr_BY",
    "fwe_simes_hochberg",
    "fwe_holm_bonferroni",
    "fwe_holm_sidak",
    "fwe_sidak",
    "fwe_bonferroni",
]

ref_subjects = [
    "sub-adult15",
    "sub-adult16",
    "sub-xp201",
    "sub-xp207",
    "sub-1",
    "sub-36",
    "sub-CTS201",
    "sub-CTS210",
]


def open_file(filename):
    with open(filename, "rb") as fi:
        return pickle.load(fi)
    return None


def get_test(
    test, confidence, reference, reference_subject, target, target_subject, fwh
):
    regexp = [
        test,
        confidence,
        "reference",
        reference,
        reference_subject,
        "target",
        target,
        target_subject,
        "fwh",
        fwh,
    ]
    regexp = "_".join(regexp)
    files = glob.glob(regexp)
    if len(files) == 0:
        return []
    [include] = files
    return include


def get_pce_exclude(
    args, df, alpha, alternative="two-sided", ratio=False, high_confidence=False
):
    """
    Return tests that passes
    """

    def ttest(series_of_struct):
        _values = [
            scipy.stats.ttest_1samp(
                next_struct["fvr"],
                popmean=next_struct["alpha"],
                alternative=alternative,
            ).pvalue
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    def ttest_ci(series_of_struct):
        _values = [
            tuple(
                scipy.stats.ttest_1samp(
                    next_struct["fvr"],
                    popmean=next_struct["alpha"],
                    alternative=alternative,
                ).confidence_interval(confidence_level=1 - alpha)
            )
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    df = df.filter((pd.col("method") == "pce"))

    if high_confidence:
        df = df.filter(pd.col("confidence") > 0.99)
    else:
        df = df.filter((0.5 <= pd.col("confidence")) & (pd.col("confidence") <= 0.999))

    indexes = [
        "reference_dataset",
        "reference_subject",
        "reference_template",
        "target_dataset",
        "target_subject",
        "target_template",
        "confidence",
        "fwhm",
        "alpha",
        "mask",
    ]

    df = df.with_columns(
        (
            (1 - pd.col("confidence")).alias("alpha"),
            (pd.col("reject") / pd.col("tests")).alias("fvr"),
        )
    )

    df = df.groupby(indexes).agg([pd.col("fvr"), pd.col("fvr").mean().alias("ratio")])

    df = df.with_columns(((pd.struct(["fvr", "alpha"]).map(ttest)).alias("pvalue")))

    df = df.with_columns((pd.col("pvalue") >= alpha).alias("success"))

    return df


def get_pce_one(
    args, df, alpha, alternative="two-sided", ratio=False, high_confidence=False
):
    """
    Return tests that passes
    """

    df = df.filter((pd.col("method") == "pce"))

    if high_confidence:
        df = df.filter(pd.col("confidence") > 0.99)
    else:
        df = df.filter((0.5 <= pd.col("confidence")) & (pd.col("confidence") <= 0.999))

    df = df.with_columns(
        (
            (1 - pd.col("confidence")).alias("alpha"),
            (pd.col("reject") / pd.col("tests")).alias("fvr"),
        )
    )

    df = df.with_columns((pd.col("fvr") <= pd.col("alpha")).alias("success"))

    return df


def get_pce_deviation(args, df):
    df = df[df["method"] == "pce"]

    indexes = ["prefix", "dataset", "subject", "confidence", "fwh", "sample_size"]
    drop = ["kth_round", "nb_round", "target"]
    print(df)
    df["alpha"] = 1 - df["confidence"]
    df["positive"] = df["reject"] / df["tests"]
    x = (
        df.drop(drop, axis=1)
        .groupby(indexes)
        .agg([np.mean, np.var, list])
        .apply(lambda t: t)
    )
    print(x)
    x["neff"] = (x["alpha"]["mean"] * (1 - x["alpha"]["mean"])) / x["positive"]["var"]
    y = x[x["positive"]["mean"] != 0]
    print("y", y)
    z = y.apply(
        lambda t: scipy.stats.norm.sf(
            t["positive"]["list"], loc=t["alpha"]["mean"], scale=1 / t["neff"]
        ),
        axis=1,
    )
    # print(z)

    print("z", z.groupby(indexes).apply(lambda t: (np.mean(t[-1]), np.std(t[-1]))))
    sys.exit(1)


def get_mct_exclude(
    args,
    df,
    alpha,
    alternative="two-sided",
    ratio=False,
    method="fwe_bonferroni",
    high_confidence=False,
):
    """
    upper bound  CI for the number of failures
    """

    def binom(series_of_struct):
        _values = [
            scipy.stats.binomtest(
                k=next_struct["fails"],
                n=next_struct["trials"],
                p=next_struct["alpha"],
                alternative="greater",
            ).pvalue
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    def binom_succ(series_of_struct):
        """
        lower bound CI for for the number of successes (should be equivalent to number of failures with p=alpha)
        """
        _values = [
            scipy.stats.binomtest(
                k=next_struct["trials"] - next_struct["fails"],
                n=next_struct["trials"],
                p=1 - next_struct["alpha"],
                alternative="less",
            ).pvalue
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    indexes = [
        "reference_dataset",
        "reference_subject",
        "reference_template",
        "target_dataset",
        "target_subject",
        "target_template",
        "confidence",
        "fwhm",
        "alpha",
        "mask",
    ]

    df = df.filter((pd.col("method") == method))

    if high_confidence:
        df = df.filter(pd.col("confidence") > 0.99)
    else:
        df = df.filter((0.5 <= pd.col("confidence")) & (pd.col("confidence") <= 0.999))

    df = df.with_columns(
        (
            (pd.col("reject") > 0).alias("fail"),
            (1 - pd.col("confidence")).alias("alpha"),
        )
    )

    df = df.groupby(indexes).agg(
        [
            (pd.col("fail").sum()).alias("fails"),
            (pd.col("fail").count()).alias("trials"),
            (pd.col("fail").mean()).alias("ratio"),
        ]
    )

    df = df.with_columns(
        (pd.struct(["fails", "trials", "alpha"]).map(binom_succ).alias("pvalue"))
    )

    df = df.with_columns((pd.col("pvalue") >= alpha).alias("success"))

    return df


def get_mct_one(
    args,
    df,
    alpha,
    alternative="two-sided",
    ratio=False,
    method="fwe_bonferroni",
    high_confidence=False,
):
    """
    Return tests that passes
    """

    df = df.filter((pd.col("method") == method))

    if high_confidence:
        df = df.filter(pd.col("confidence") > 0.99)
    else:
        df = df.filter((0.5 <= pd.col("confidence")) & (pd.col("confidence") <= 0.999))

    df = df.with_columns(
        (
            (pd.col("reject") > 0).alias("fail"),
            (1 - pd.col("confidence")).alias("alpha"),
        )
    )

    df = df.with_columns((pd.col("reject") <= 0).alias("success"))

    return df


def plot_test_exclude(tests, ratio=False, verbose=False):
    if ratio:
        colors = "RdYlGn_r"
        zmin = 0
        zmax = 1
    else:
        colors = ["#E6020D", "#007A0E"]
        colors = ["#d60000", "#006b0c"]
        zmin = 0
        zmax = 1

    subjects = (
        tests[0]
        .collect()
        .select(pd.col("reference_subject"))
        .unique()
        .sort(by=["reference_subject"])
        .to_dict(as_series=False)["reference_subject"]
    )
    cols = len(tests)
    rows = len(subjects)

    test_fig = make_subplots(
        rows=rows,
        cols=cols,
        column_titles=["RR", "RS", "RR+RS"],
        row_titles=subjects,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="FWHM (mm)",
        y_title="Confidence level",
        vertical_spacing=0.005,
        horizontal_spacing=0.005,
    )

    for col, test in enumerate(tests, start=1):
        for row, subject in enumerate(subjects, start=1):
            for a in test_fig["layout"]["annotations"]:
                a["textangle"] = 0

            pce_subject = (
                test.filter(pd.col("reference_subject") == subject)
                .sort(by=["confidence", "fwhm"], descending=[False, False])
                .collect()
            )

            if ratio:
                pivot = pce_subject.pivot(
                    index=["confidence"], columns=["fwhm"], values="ratio"
                )
            else:
                pivot = pce_subject.pivot(
                    index=["confidence"], columns=["fwhm"], values="success"
                )

            confidences = pivot["confidence"].to_numpy()
            fwhms = pivot.columns[1:]
            z = pivot.to_numpy()[..., 1:]

            if verbose:
                print(subject)
                print("x", confidences.shape)
                print(confidences)
                print("y", len(fwhms))
                print(fwhms)
                print("z", z.shape)
                print(pivot)

            im = px.imshow(
                z,
                x=[str(int(float(f))) for f in fwhms],
                y=[str(f) for f in confidences],
                zmin=zmin,
                zmax=zmax,
                color_continuous_scale=colors,
                origin="lower",
            )

            test_fig.add_trace(im.data[0], row=row, col=col)

    test_fig.for_each_xaxis(lambda xaxis: xaxis.tickfont.update(size=7))
    test_fig.for_each_yaxis(lambda yaxis: yaxis.tickfont.update(size=7))

    test_fig.update_xaxes(tickangle=0)
    test_fig.update_layout(coloraxis=dict(colorscale=colors))
    test_fig.update_coloraxes(cmin=0, cmax=1)
    test_fig.update_layout(margin=dict(t=25, b=40, l=40, r=60))

    test_fig.update_annotations(font=dict(size=12))
    test_fig.layout["annotations"][-1]["xshift"] = -15
    test_fig.layout["annotations"][-2]["yshift"] = -10

    if not args.ratio:
        test_fig.update_traces(showlegend=False)
        test_fig.update_coloraxes(showscale=False)
    else:
        test_fig.update_layout(coloraxis_colorbar_x=1.05)

    test_fig["layout"]["annotations"][-1]["textangle"] = -90
    test_fig.update_layout(font_family="Serif")

    return test_fig


def get_parameters_heatmap_subject(test, confidence, fwhm, subjects, verbose):
    cell = test.filter(
        (pd.col("confidence") == confidence) & (pd.col("fwhm") == fwhm)
    ).sort(by=["confidence", "fwhm", "reference_subject", "target_subject"])

    pivot_index = ["reference_subject"]
    pivot_columns = ["target_subject"]
    pivot = cell.pivot(index=pivot_index, columns=pivot_columns, values="success")

    x = pivot["reference_subject"].to_numpy()
    y = pivot.columns[1:]
    z = pivot.with_columns((pd.col(subject).cast(pd.Int8) for subject in subjects))

    z = np.rot90(z.to_numpy()[..., 1:])

    if verbose:
        print("=" * 30)
        print(f"Confidence: {confidence}, FWHM: {fwhm}")
        print(pivot)
        print("x", x.shape)
        print(x)
        print("y", len(y))
        print(y)
        print("z", z.shape)
        print(z)

    return (x, y, z)


def get_parameters_heatmap_template(test, confidence, fwhm, templates, verbose):
    cell = test.filter(
        (pd.col("confidence") == confidence) & (pd.col("fwhm") == fwhm)
    ).sort(by=["confidence", "fwhm", "reference_subject", "target_template"])

    pivot_index = ["reference_subject"]
    pivot_columns = ["target_template"]
    pivot = cell.pivot(index=pivot_index, columns=pivot_columns, values="success")

    x = pivot.columns[1:]
    y = pivot["reference_subject"].to_numpy()
    z = pivot.with_columns((pd.col(template).cast(pd.Int8) for template in templates))
    z = z.to_numpy()[..., 1:]

    if verbose:
        print("=" * 30)
        print(f"Confidence: {confidence}, FWHM: {fwhm}")
        print(pivot)
        print("x", len(x))
        print(x)
        print("y", y.shape)
        print(y)
        print("z", z.shape)
        print(z)

    return (x, y, z)


def plot_test_template(tests, verbose=False):
    colors = ["#d60000", "#006b0c"]
    zmin = 0
    zmax = 1

    tests_label = ["   RR", "   RS", "RR+RS"]
    tests = [test.collect() for test in tests]
    nb_tests = len(tests)
    test = tests[0]

    subjects = (
        test.select(pd.col("reference_subject"))
        .unique()
        .sort(by=["reference_subject"])
        .to_dict(as_series=False)["reference_subject"]
    )

    templates = test["target_template"].unique().sort().to_numpy()

    rows = templates.size
    cols = len(subjects)

    column_titles = subjects

    row_titles = []

    for label in tests_label[:nb_tests]:
        for i, t in enumerate(templates, start=1):
            title = str(int(t.split("Noised")[-1]) / 10) + "%"
            title = f"{title:>6}"
            row_titles.append(title)

    specs = [[{} for _ in range(cols)] for _ in range(rows * nb_tests)]

    for r in [i * rows - 1 for i in range(1, nb_tests)]:
        for cell in specs[r]:
            cell["b"] = 0.003

    test_fig = make_subplots(
        rows=rows * nb_tests,
        cols=cols,
        column_titles=column_titles,
        row_titles=row_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="FWHM (mm)",
        y_title="Alpha threshold",
        vertical_spacing=0.00001,
        horizontal_spacing=0.001,
        specs=specs,
    )

    for i, test in enumerate(tests, start=0):
        for row, template in enumerate(templates, start=1):
            for col, subject in enumerate(subjects, start=1):
                for a in test_fig["layout"]["annotations"]:
                    a["textangle"] = 0

                pce_subject = test.filter(
                    (pd.col("reference_subject") == subject)
                    & (pd.col("target_template") == template)
                ).sort(by=["confidence", "fwhm"], descending=[False, False])

                pivot = pce_subject.pivot(
                    index=["confidence"], columns=["fwhm"], values="success"
                )

                confidences = pivot["confidence"].to_numpy()
                fwhms = pivot.columns[1:]
                z = pivot.to_numpy()[..., 1:]

                if verbose:
                    print(subject)
                    print("x", confidences.shape)
                    print(confidences)
                    print("y", len(fwhms))
                    print(fwhms)
                    print("z", z.shape)
                    print(pivot)

                im = px.imshow(
                    z,
                    x=[str(int(float(f))) for f in fwhms],
                    y=[str(f) for f in confidences],
                    zmin=zmin,
                    zmax=zmax,
                    color_continuous_scale=colors,
                    origin="lower",
                )

                test_fig.add_trace(im.data[0], row=row + i * rows, col=col)

        test_fig.for_each_xaxis(lambda xaxis: xaxis.tickfont.update(size=7))
        test_fig.for_each_yaxis(lambda yaxis: yaxis.tickfont.update(size=3))

        test_fig.update_xaxes(tickangle=0)
        test_fig.update_layout(coloraxis=dict(colorscale=colors))
        test_fig.update_coloraxes(cmin=0, cmax=1)
        test_fig.update_layout(margin=dict(t=25, b=30, l=30, r=70))

        test_fig.update_annotations(font=dict(size=10))
        test_fig.layout["annotations"][-1]["xshift"] = -10
        test_fig.layout["annotations"][-2]["yshift"] = -10

        if not args.ratio:
            test_fig.update_traces(showlegend=False)
            test_fig.update_coloraxes(showscale=False)
        else:
            test_fig.update_layout(coloraxis_colorbar_x=1.05)

        test_fig["layout"]["annotations"][-1]["textangle"] = -90

        test_fig.for_each_xaxis(
            lambda x: x.update(
                tickmode="array",
                tickvals=[1, 5, 10, 15, 19],
                ticktext=["0", "5", "10", "15", "20"],
            )
        )
        test_fig.for_each_yaxis(
            lambda y: y.update(
                tickmode="array",
                tickvals=[0.99, 0.8, 0.6],
                ticktext=["0.005", "0.25", "0.50"],
            )
        )
        test_fig.update_layout(font_family="Serif")

    return test_fig


def plot_test_versions(tests, verbose=False):
    colors = ["#d60000", "#006b0c"]
    zmin = 0
    zmax = 1

    tests_label = ["   RR", "   RS", "RR+RS"]
    tests = [test.collect() for test in tests]
    nb_tests = len(tests)
    test = tests[0]

    subjects = (
        test.select(pd.col("reference_subject"))
        .unique()
        .sort(by=["reference_subject"])
        .to_dict(as_series=False)["reference_subject"]
    )

    versions = test["target_version"].unique().sort().to_numpy()

    rows = versions.size
    cols = len(subjects)

    column_titles = subjects

    row_titles = []

    for label in tests_label[:nb_tests]:
        for i, t in enumerate(versions, start=1):
            title = f"{t:>6}"
            row_titles.append(title)

    specs = [[{} for _ in range(cols)] for _ in range(rows * nb_tests)]

    for r in [i * rows - 1 for i in range(1, nb_tests)]:
        for cell in specs[r]:
            cell["b"] = 0.003

    test_fig = make_subplots(
        rows=rows * nb_tests,
        cols=cols,
        column_titles=column_titles,
        row_titles=row_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="FWHM (mm)",
        y_title="Alpha threshold",
        vertical_spacing=0.001,
        horizontal_spacing=0.001,
        specs=specs,
    )

    for i, test in enumerate(tests, start=0):
        for row, version in enumerate(versions, start=1):
            for col, subject in enumerate(subjects, start=1):
                for a in test_fig["layout"]["annotations"]:
                    a["textangle"] = 0

                pce_subject = test.filter(
                    (pd.col("reference_subject") == subject)
                    & (pd.col("target_version") == version)
                ).sort(by=["confidence", "fwhm"], descending=[False, False])

                pivot = pce_subject.pivot(
                    index=["confidence"], columns=["fwhm"], values="success"
                )

                confidences = pivot["confidence"].to_numpy()
                fwhms = pivot.columns[1:]
                z = pivot.to_numpy()[..., 1:]

                if verbose:
                    print(subject)
                    print("x", confidences.shape)
                    print(confidences)
                    print("y", len(fwhms))
                    print(fwhms)
                    print("z", z.shape)
                    print(pivot)

                im = px.imshow(
                    z,
                    x=[str(int(float(f))) for f in fwhms],
                    y=[str(f) for f in confidences],
                    zmin=zmin,
                    zmax=zmax,
                    color_continuous_scale=colors,
                    origin="lower",
                )

                test_fig.add_trace(im.data[0], row=row + i * rows, col=col)

        test_fig.for_each_xaxis(lambda xaxis: xaxis.tickfont.update(size=7))
        test_fig.for_each_yaxis(lambda yaxis: yaxis.tickfont.update(size=3))

        test_fig.update_xaxes(tickangle=0)
        test_fig.update_layout(coloraxis=dict(colorscale=colors))
        test_fig.update_coloraxes(cmin=0, cmax=1)
        test_fig.update_layout(margin=dict(t=25, b=30, l=30, r=70))

        test_fig.update_annotations(font=dict(size=10))
        test_fig.layout["annotations"][-1]["xshift"] = -10
        test_fig.layout["annotations"][-2]["yshift"] = -10

        if not args.ratio:
            test_fig.update_traces(showlegend=False)
            test_fig.update_coloraxes(showscale=False)
        else:
            test_fig.update_layout(coloraxis_colorbar_x=1.05)

        test_fig["layout"]["annotations"][-1]["textangle"] = -90

        test_fig.for_each_xaxis(
            lambda x: x.update(
                tickmode="array",
                tickvals=[1, 5, 10, 15, 19],
                ticktext=["0", "5", "10", "15", "20"],
            )
        )
        test_fig.for_each_yaxis(
            lambda y: y.update(
                tickmode="array",
                tickvals=[0.995, 0.95, 0.85, 0.75, 0.65, 0.55],
                ticktext=["0.005", "0.05", "0.15", "0.25", "0.35", "0.45"],
            )
        )
        test_fig.update_layout(font_family="Serif")

    return test_fig


def plot_test_one(labels, tests, ratio=False, verbose=False):
    if ratio:
        colors = "RdYlGn_r"
        zmin = 0
        zmax = 1
    else:
        colors = ["#d63020", "#007722"]
        zmin = 0
        zmax = 1

    tests_label = ["   RR", "   RS", "RR+RS"]
    tests = [test.collect() for test in tests]
    nb_tests = len(tests)
    test = tests[0]

    subjects = (
        test.select(pd.col("reference_subject"))
        .unique()
        .sort(by=["reference_subject"])
        .to_dict(as_series=False)["reference_subject"]
    )
    confidences = test["confidence"].unique().sort(descending=True).to_numpy()

    fwhms = test["fwhm"].unique().sort().to_numpy()

    if args.no_x_title:
        column_titles = ["" for f in fwhms]
        xtitle = ""
    else:
        column_titles = [str(int(f)) for f in fwhms]
        xtitle = "FWHM (mm)"

    row_titles = [f"{1-c:.3f}" for c in confidences]

    figs = []

    i = 0
    for label, test in zip(labels, tests):
        i += 1
        print(label)

        rows = confidences.size
        cols = fwhms.size
        print(rows, cols)

        pce_fig = make_subplots(
            rows=rows,
            cols=cols,
            column_titles=column_titles,
            row_titles=row_titles,
            shared_xaxes=True,
            shared_yaxes=True,
            x_title=xtitle,
            y_title="Alpha threhold",
            vertical_spacing=0,
            horizontal_spacing=0,
        )

        for a in pce_fig["layout"]["annotations"]:
            a["textangle"] = 0

        econfidences = enumerate(confidences, start=1)
        efwhms = enumerate(fwhms, start=1)

        _iterator = tqdm.tqdm(
            itertools.product(econfidences, efwhms), total=rows * cols
        )
        for (row, confidence), (col, fwhm) in _iterator:
            (x, y, z) = get_parameters_heatmap_subject(
                test, confidence, fwhm, subjects, verbose=args.verbose
            )

            diag = np.rot90(np.eye(z.shape[0], dtype=bool))
            z_transformed = z

            if verbose:
                print(z_transformed)

            im = px.imshow(
                z_transformed,
                x=[str(i) for i in x],
                y=[str(i) for i in y],
                zmin=zmin,
                zmax=zmax,
                aspect="equal",
                color_continuous_scale=colors,
                origin="upper",
            )

            pce_fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=len(x) - 1,
                y0=0,
                x1=0,
                y1=len(y) - 1,
                opacity=0.25,
                line=dict(color="black"),
                row=row,
                col=col,
            )

            pce_fig.add_trace(im.data[0], row=row, col=col)

            if args.no_x_title:
                bottom, top = 5, 5
            else:
                bottom, top = 35, 5

            pce_fig.update_layout(coloraxis=dict(colorscale=colors))
            pce_fig.update_coloraxes(cmin=0, cmax=1)
            pce_fig.update_layout(margin=dict(t=top, b=bottom, r=23, l=25))
            pce_fig.layout["annotations"][-1]["xshift"] = -5
            pce_fig.layout["annotations"][-2]["yshift"] = -12

            if not args.ratio:
                pce_fig.update_traces(showlegend=False)
                pce_fig.update_coloraxes(showscale=False)
            else:
                pce_fig.update_layout(coloraxis_colorbar_x=1.05)

            pce_fig["layout"]["annotations"][-1]["textangle"] = -90

        pce_fig.update_xaxes(showticklabels=False)
        pce_fig.update_yaxes(showticklabels=False)
        pce_fig.update_layout_images(
            xaxis_showticklabels=False, yaxis_showticklabels=False, font=dict(size=12)
        )
        pce_fig.update_layout(
            font=dict(size=12), annotations=[dict(font=dict(size=12))]
        )
        pce_fig.update_layout(font_family="Serif")

        for a in pce_fig["layout"]["annotations"]:
            if a["text"] in column_titles:
                a["y"] = -0.033

        # print(pce_fig)

    return pce_fig


def plotly_backend_exclude(args, pces, mcts, show, no_pce, no_mct, ratio=False):
    if not no_pce:
        pce_fig = plot_test_exclude(pces, ratio, args.verbose)
    if not no_mct:
        mct_fig = plot_test_exclude(mcts, ratio, args.verbose)

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()

    ext = "_ratio" if args.ratio else ""

    if not no_pce:
        pce_fig.write_image(f"{args.test}_pce{ext}.pdf")
    if not no_mct:
        mct_fig.write_image(f"{args.test}_mct{ext}_{args.mct_method}.pdf")


def plotly_backend_one(args, pces, mcts, show, no_pce, no_mct, ratio=False):
    labels = ["RR", "RS", "RR+RS"]
    if not no_pce:
        print("PCE")
        if args.template:
            pce_fig = plot_test_template(pces, args.verbose)
        elif args.versions:
            pce_fig = plot_test_versions(pces, args.verbose)
        else:
            pce_fig = plot_test_one(labels, pces, ratio, args.verbose)
    if not no_mct:
        print(f"MCT {args.mct_method}")
        if args.template:
            mct_fig = plot_test_template(mcts, args.verbose)
        elif args.versions:
            mct_fig = plot_test_versions(mcts, args.verbose)
        else:
            mct_fig = plot_test_one(labels, mcts, ratio, args.verbose)

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()

    ext = ("_ratio" if args.ratio else "") + ("_template" if args.template else "")
    dim = dict(width=720 * 3, height=720) if args.template else dict()
    dim = dict()
    if not no_pce:
        pce_fig.write_image(f"{args.test}_pce_{ext}.pdf", scale=1, **dim)
    if not no_mct:
        mct_fig.write_image(
            f"{args.test}_mct_{args.mct_method}_{ext}.pdf", scale=1, **dim
        )


def get_optimum(df):
    df = df.collect()
    confidences = df["confidence"].unique().sort()
    print("Local optimum per alpha threshold")
    (alpha_star, fwhm_star) = (-1, -1)

    succ = df.groupby(["confidence", "fwhm"]).agg(
        [pd.col("success").sum().alias("successes")]
    )
    succ = succ.with_columns((1 - pd.col("confidence")).alias("alpha"))
    max = succ.select(pd.col("successes").max())

    for confidence in confidences:
        dfc = succ.filter(pd.col("confidence") == confidence)
        argmax_df = (
            dfc.filter(pd.col("successes") == max)
            .sort(by=["confidence", "fwhm"])
            .select([pd.col("alpha"), pd.col("fwhm")])
        )
        if argmax_df.height != 0:
            (alpha_star, fwhm_star) = argmax_df.to_numpy().min(axis=0)
            print(f" * {alpha_star:.6f}, {fwhm_star}")

    return (alpha_star, fwhm_star)


def parse_dataframe(dfs, test, **kwds):
    return list(map(lambda df: test(args, df, **kwds), dfs))


def get_optimum_test(references, test_name, tests, ext):
    for reference, test in zip(references, tests):
        print("=" * 30)
        print(reference)
        (alpha_star, fwh_star) = get_optimum(test)
        print(f"{test_name} alpha*={alpha_star:.6f}, fwh*={fwh_star}")
        name = reference.replace(os.path.sep, "_")
        test.collect().to_pandas().to_csv(f"{args.test}_{name}_{test_name}{ext}.csv")


def get_version(path):
    """
    retrieve fmriprep version in the path name
    TODO: add fmriprep version in the dict
    """
    return path.split("target")[1].split("_")[1].split("-")[1]


def get_reference(reference):
    paths = glob.glob(f"{reference}/*.pkl")
    ldf = []
    for path in tqdm.tqdm(paths):
        with open(path, "rb") as fib:
            pkl = pickle.load(fib)
            df = pd.DataFrame(pkl).lazy()
            version = get_version(path)
            df = df.with_columns(pd.Series(name="target_version", values=[version]))
            ldf.append(df)
    return pd.concat(ldf)


def memoize(arg, fun):
    _raw_hash = hashlib.md5(arg.encode("utf-8")).hexdigest()
    _mem_file = f"{_raw_hash}.pkl"
    if os.path.exists(_mem_file):
        with open(_mem_file, "rb") as fi:
            return pickle.load(fi)
    else:
        res = fun(arg)
        with open(_mem_file, "wb") as fo:
            pickle.dump(res, fo)
        return res


def get_references(references):
    dfs = []
    for reference in references:
        ldf = get_reference(reference)
        ldf = ldf.with_columns((pd.Series(name="prefix", values=[reference])))
        dfs.append(ldf)
    return dfs


def plot_exclude(args):
    # pd.Config().set_tbl_rows(1500)

    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    ext = "_ratio" if args.ratio else ""

    dfs = get_references(references)

    if not args.no_pce:
        pce_tests = parse_dataframe(
            dfs,
            get_pce_exclude,
            alpha=alpha,
            alternative="greater",
            ratio=args.ratio,
            high_confidence=args.high_confidence,
        )
        get_optimum_test(references, "pce", pce_tests, ext)
    else:
        pce_tests = []

    if not args.no_mct:
        mct_tests = parse_dataframe(
            dfs,
            get_mct_exclude,
            alpha=alpha,
            alternative="greater",
            ratio=args.ratio,
            method=args.mct_method,
            high_confidence=args.high_confidence,
        )
        get_optimum_test(references, "mct", mct_tests, ext)
    else:
        mct_tests = []

    plotly_backend_exclude(
        args,
        pce_tests,
        mct_tests,
        show,
        no_pce=args.no_pce,
        no_mct=args.no_mct,
        ratio=args.ratio,
    )


def plot_one(args):
    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    dfs = get_references(references)
    if not args.no_pce:
        pce_tests = parse_dataframe(
            dfs,
            get_pce_one,
            alpha=alpha,
            alternative="greater",
            ratio=args.ratio,
            high_confidence=args.high_confidence,
        )
    else:
        pce_tests = []

    if not args.no_mct:
        mct_tests = parse_dataframe(
            dfs,
            get_mct_one,
            alpha=alpha,
            alternative="greater",
            ratio=args.ratio,
            method=args.mct_method,
            high_confidence=args.high_confidence,
        )
    else:
        mct_tests = []

    if args.ieee:
        plotly_backend_exclude(
            args,
            pce_tests,
            mct_tests,
            show,
            no_pce=args.no_pce,
            no_mct=args.no_mct,
            ratio=args.ratio,
        )
    else:
        plotly_backend_one(
            args,
            pce_tests,
            mct_tests,
            show,
            no_pce=args.no_pce,
            no_mct=args.no_mct,
            ratio=args.ratio,
        )


def plot_deviation_exclude(args):
    pd.set_option("display.max_rows", None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('max_colwidth', -1)

    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    ext = "_ratio" if args.ratio else ""

    dfs = []

    for reference in references:
        paths = glob.glob(f"{reference}/*.pkl")
        ldf = []
        for path in paths:
            with open(path, "rb") as fib:
                pkl = pickle.load(fib)
                df = pd.DataFrame(pkl)
                df.insert(0, "prefix", reference)
                ldf.append(df)
        dfs.append(pd.concat(ldf))

    pce_tests, mct_tests = [], []
    for df in dfs:
        pce_tests.append(get_pce_deviation(args, df))

    for reference, pce_test in zip(references, pce_tests):
        print("=" * 30)
        print(reference)
        (alpha_star, fwh_star) = get_optimum(pce_test)
        print(f"pce alpha*={alpha_star}, fwh*={fwh_star}")
        name = reference.replace(os.path.sep, "_")
        pce_test.to_csv(f"{args.test}_{name}_pce{ext}.csv")

    plotly_backend(
        args,
        pce_tests,
        mct_tests,
        show,
        no_pce=args.no_pce,
        no_mct=args.no_mct,
        ratio=args.ratio,
    )


def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--reference", required=True, nargs="+")
    parser.add_argument("--test", choices=tests_name.keys(), required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--print-args", action="store_true", help="Show passed args")
    parser.add_argument(
        "--meta-alpha", default=0.05, type=float, help="Alpha on permutation test"
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--no-pce", action="store_true", help="Do not show PCE")
    parser.add_argument(
        "--no-x-title", action="store_true", default=False, help="x title"
    )
    parser.add_argument("--no-mct", action="store_true", help="Do not show MCT")
    parser.add_argument("--show-nan", action="store_true", help="Show NaN")
    parser.add_argument("--ratio", action="store_true", help="Print ratio")
    parser.add_argument(
        "--deviation", action="store_true", help="Compute deviation statistics"
    )
    parser.add_argument("--mct-method", default="fwe_bonferroni")
    parser.add_argument(
        "--high-confidence",
        action="store_true",
        help="show confidence level 0.999 0.9999 0.99999 0.999999",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    parser.add_argument(
        "--template", action="store_true", help="compare agains template"
    )
    parser.add_argument(
        "--versions", action="store_true", help="compare against versions"
    )
    parser.add_argument("--ieee", action="store_true")
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()
    if args.print_args:
        print(args)

    elif args.test == "exclude":
        if args.deviation:
            plot_deviation_exclude(args)
        else:
            plot_exclude(args)
    elif args.test == "one":
        plot_one(args)
