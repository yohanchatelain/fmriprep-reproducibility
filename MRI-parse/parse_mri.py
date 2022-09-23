import pandas as pd
import argparse
import sys
import pickle
import os
import plotly.express as px
import glob
import plotly.graph_objects as go

tests_name = {
    'exclude': 'all-exclude',
    'include': 'all-include',
    'one': 'one'
}

methods_name = [
    'pce',
    'fdr_storey',
    'fdr_BH',
    'fdr_BY',
    'fwe_holm_bonferroni',
    'fwe_bonferroni'
]

ref_subjects = [
    'sub-adult15',
    'sub-adult16',
    'sub-xp201',
    'sub-xp207',
    'sub-1',
    'sub-CTS201',
    'sub-CTS210'
]


def open_file(filename):
    with open(filename, 'rb') as fi:
        return pickle.load(fi)
    return None


def get_test(directory, test, confidence, reference_subject, target_subject=''):
    test_name = tests_name[test]
    if glob.glob(
            f'{directory}{os.sep}{test_name}*{confidence}_reference_*_{reference_subject}_target*{target_subject}*.pkl') == []:
        return []
    [exclude] = glob.glob(
        f'{directory}{os.sep}{test_name}*{confidence}_reference_*_{reference_subject}_target*{target_subject}*.pkl')
    return open_file(exclude)


def plot(directory, test, confidence, subjects, show=False):

    dl = []

    for subject in subjects:
        values = get_test(directory, test, confidence, subject)
        methods = dict(zip(methods_name, [[] for _ in methods_name]))
        for value in values:
            methods[value['method']].append(value['fvr'])

        d = pd.DataFrame.from_dict(methods)
        d.insert(0, 'subject', subject)
        dl.append(d)

    d = pd.concat(dl)
    fig = px.violin(d, color='subject', box=True)
    fig.add_hline(1.0 - float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio')
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test}')
    if show:
        fig.show()
    fig.write_image(f'{test}_{confidence}.jpg')


def plot_one(directory, test, confidence, subjects, show=False):

    dl = []

    for target in ref_subjects:
        for subject in subjects:
            values = get_test(directory, test, confidence, subject, target)
            methods = dict(zip(methods_name, [[] for _ in methods_name]))
            for value in values:
                methods[value['method']].append(value['fvr'])

            d = pd.DataFrame.from_dict(methods)
            d.insert(0, 'subject', target)
            dl.append(d)

    d = pd.concat(dl)
    fig = px.violin(d, color='subject', box=True)
    fig.add_hline(1.0 - float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio')
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test}')
    if show:
        fig.show()
    fig.write_image(f'{test}_{confidence}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--directory', default='mri_pickle')
    parser.add_argument('--test', choices=tests_name.keys(), required=True)
    parser.add_argument('--subjects', required=True, nargs='+')
    parser.add_argument('--confidence', required=True)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--print-args', action='store_true',
                        help="Show passed args")
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    if args.print_args:
        print(args)

    if args.test == 'one':
        plot_one(args.directory, args.test,
                 args.confidence, args.subjects, args.show)
    else:
        plot(args.directory, args.test, args.confidence, args.subjects, args.show)
