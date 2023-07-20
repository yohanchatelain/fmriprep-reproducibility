import pandas as pd
import argparse
import os


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


def plot_experiment_ieee(df: pd.DataFrame):
    print(df)
    # 1- group by reference_perturbation
    # 2- group by reference_dataset, reference_subject
    # 3- split between (reference_dataset, reference_subject) == (target_dataset, target_subject)
    # 4- Compute ratio successes

    df.groupby(["reference_perturbation"], inplace=True)
    print(df)


_plot_experiment = {"ieee": plot_experiment_ieee}


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
