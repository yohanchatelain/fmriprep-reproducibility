import argparse
from matplotlib import category

import pandas as pd
import plotly
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import dataset_mapping as dsmap

pio.kaleido.scope.mathjax = None


def show_named_plotly_colours():
    """
    function to display to user the colours to match plotly's named
    css colours.

    Reference:
        #https://community.plotly.com/t/plotly-colours-list/11730/3

    Returns:
        plotly dataframe with cell colour to match named colour name

    """
    s = """
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        """
    li = s.split(",")
    li = [line.replace("\n", "") for line in li]
    li = [line.replace(" ", "") for line in li]

    import plotly.graph_objects as go

    df = pd.DataFrame.from_dict({"colour": li})
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Plotly Named CSS colours"],
                    line_color="black",
                    fill_color="white",
                    align="center",
                    font=dict(color="black", size=14),
                ),
                cells=dict(
                    values=[df.colour],
                    line_color=[df.colour],
                    fill_color=[df.colour],
                    align="center",
                    font=dict(color="black", size=11),
                ),
            )
        ]
    )

    fig.show()


def parse_input(args):
    df = pd.read_csv(args.input)
    df = df.drop("Unnamed: 0", axis=1)

    df_modes = []

    for mode in args.perturbation_modes:
        df_mode = df[df["prefix"] == mode]
        df_mode.sort_values(by=["fwh", "stat", "subject"], axis=0, inplace=True)
        df_modes.append(df_mode)

    pd.set_option("display.max_rows", 600)

    return df_modes


def get_number_statistics_to_plot(args):
    _stats = [args.mean, args.std, args.sig]
    return sum(map(lambda x: x != "no", _stats))


def get_row_titles(args):
    row_titles = []
    if args.mean != "no":
        log = " (log)" if args.mean == "log" else ""
        row_titles += [f"Mean{log}"]
    if args.std != "no":
        log = " (log)" if args.std == "log" else ""
        row_titles += [f"Standard deviation{log}"]
    if args.sig != "no":
        row_titles += ["Significant bits"]

    return row_titles


def create_figure(args):
    """
    create subplots with n rows and m columns
    n: number of statistics (mean, std, sig)
    m: number of perturbations modes (rr, rs, rr+rs)
    """

    nrows = get_number_statistics_to_plot(args)
    ncols = len(args.perturbation_modes)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=args.vertical_spacing,
        horizontal_spacing=args.horizontal_spacing,
        column_titles=args.column_titles,
        # row_titles=get_row_titles(args),
        x_title="FWHM (mm)",
        y_title="Significant bits",
    )

    return fig


def create_dataset_subject_map(df_modes):
    _ds = dict()
    for df_mode in df_modes:
        ds = df_mode[["dataset", "subject"]]
        for row in ds.iterrows():
            dataset = row[1]["dataset"]
            subject = row[1]["subject"]
            _ds[dataset] = _ds.get(dataset, {}) | {subject: ""}
    return _ds


def get_statistics(args):
    stats = []
    if args.mean != "no":
        stat = dict(name="mean", log_y=args.mean == "log", range_y=[-10, 6000])
        stats.append(stat)
    if args.std != "no":
        stat = dict(name="std", log_y=args.std == "log", range_y=[-3, 60])
        stats.append(stat)
    if args.sig != "no":
        stat = dict(name="sig", log_y=args.sig == "log", range_y=[-3, 11])
        stats.append(stat)
    return stats


def plot(args):
    df_modes = parse_input(args)

    fig = create_figure(args)

    ds_map = create_dataset_subject_map(df_modes)
    ds_index_map = dsmap.index(ds_map)
    labels = {_dict["subject"]: str(i) for i, _dict in ds_index_map.items()}
    category_order = {"subject": list(labels.keys())}

    statistics = get_statistics(args)

    for i, stat in enumerate(statistics, start=1):
        for j, df_mode in enumerate(df_modes, start=1):
            sc = px.scatter(
                df_mode[df_mode.stat == stat["name"]],
                x="fwh",
                y="mean",
                color="subject",
                category_orders=category_order,
                log_y=stat["log_y"],
                text="dataset",
            )
            sc.update_layout(showlegend=False)
            sc.update_yaxes(range=stat["range_y"])
            if stat["log_y"]:
                sc.update_yaxes(
                    type="log",
                    showexponent="all",
                    dtick="D2",
                    exponentformat="power",
                    row=i,
                    col=j,
                )
            fig.add_traces(sc.data, rows=i, cols=j)

    fig.update_traces(marker=dict(size=args.marker_size), mode="lines+markers")
    fig.for_each_trace(
        lambda t: t.update(
            name=str(dsmap.get_index(ds_map, dataset=t.text[0], subject=t.name))
        )
    )

    nlabels = len(labels)
    for d in fig.data[nlabels:]:
        d["showlegend"] = False

    fig.update_layout(
        font=dict(size=args.font_size - 2),
        legend=dict(
            title="Subjects",
            orientation="h",
            yanchor="bottom",
            xanchor="center",
            x=0.5,
            y=-0.3,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=10, b=30, t=30),
    )

    fig["layout"]["annotations"][-1]["xshift"] += 20
    fig["layout"]["annotations"][-2]["yshift"] -= 5

    fig.update_annotations(font=dict(size=args.font_size))
    fig.update_layout(font_family="Serif")

    if args.verbose:
        print(fig)

    fig.write_image(args.output, scale=args.output_scale)

    if args.show:
        fig.show()


def parse_args():
    parser = argparse.ArgumentParser("plot statisticts")
    parser.add_argument(
        "--input", default="stats.csv", help="statistics inputs to plot"
    )
    parser.add_argument(
        "--mean", choices=["no", "yes", "log"], default="no", help="plot mean value"
    )
    parser.add_argument(
        "--std", choices=["no", "yes", "log"], default="no", help="plot std value"
    )
    parser.add_argument(
        "--sig",
        choices=["no", "yes"],
        default="yes",
        help="plot sig value",
    )
    parser.add_argument(
        "--perturbation-modes",
        nargs="+",
        default=["rr", "rs", "rr.rs"],
        choices=["rr", "rs", "rr.rs"],
        help="Perturbations modes to plot",
    )
    parser.add_argument(
        "--vertical-spacing",
        type=float,
        default=0.01,
        help="vertical spacing between subplots",
    )
    parser.add_argument(
        "--horizontal-spacing",
        type=float,
        default=0.01,
        help="horizontal spacing between subplots",
    )
    parser.add_argument(
        "--column-titles",
        nargs="+",
        default=["RR", "RS", "RR+RS"],
        help="Subplots column titles",
    )
    parser.add_argument("--marker-size", default=3, type=int, help="marker size")
    parser.add_argument("--font-size", default=10, type=int, help="font size")
    parser.add_argument("--output", default="stats.pdf", help="output name")
    parser.add_argument("--show", action="store_true", help="show figure")
    parser.add_argument(
        "--output-scale", default=10, type=int, help="output scale factor"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot(args)


if __name__ == "__main__":
    main()
