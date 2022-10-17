import pandas as pd
import argparse
import sys
import pickle
import os
import plotly.express as px
import glob
import plotly.graph_objects as go
import scipy.stats

tests_name = {
    'exclude': 'all-exclude',
    'include': 'all-include',
    'one': 'one',
    'inter': 'one'
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


def get_test(directory, test, confidence, reference_subject, target_subject='', fwh='', verbose=False):
    test_name = tests_name[test]
    fwh = f'fwh_{fwh}' if fwh else ''
    target = f'target*{target_subject}' if target_subject else ''
    regexp = f'{directory}{os.sep}{test_name}*{confidence}_reference_*_{reference_subject}*{target}*{fwh}.pkl'
    files = glob.glob(regexp)
    print(regexp)
    print(files)
    if files  == []:
        return []
    [exclude] = files
    return open_file(exclude)

from statistics import NormalDist

def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

def plot_violin(args):

    directory = args.directory
    test = args.test
    confidence = args.confidence
    subjects = args.subjects
    show = args.show
    fwh = args.fwh
    
    dl = []

    for subject in subjects:
        values = get_test(directory, test, confidence, subject, fwh=fwh)
        methods = dict(zip(methods_name, [[] for _ in methods_name]))
        for value in values:
            if value['method'] in methods:
                methods[value['method']].append(value['fvr'])

        d = pd.DataFrame.from_dict(methods)
        d.insert(0, 'subject', subject)
        dl.append(d)

    d = pd.concat(dl)
    d.to_pickle('test.pkl')
    fig = px.violin(d, color='subject', box=True)
    fig.add_hline(1.0 - float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio')
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test}')
    if show:
        fig.show()
    fig.write_image(f'{test}_{confidence}.jpg')


def plot_box(args):

    directory = args.directory
    test = args.test
    confidence = args.confidence
    subjects = args.subjects
    show = args.show
    fwh = args.fwh
    
    dl = []

    __subject = []
    __fvr = []
    __method = []
    
    
    for subject in subjects:
        values = get_test(directory, test, confidence, subject, fwh=fwh)
        methods = dict(zip(methods_name, [[] for _ in methods_name]))
        for value in values:
            if value['method'] in methods:
                __subject.append(subject)
                __fvr.append(value['fvr'])
                __method.append(value['method'])


    d = pd.DataFrame.from_dict({'subject':__subject,
                                'ratio':__fvr,
                                'method':__method})
    d.to_pickle('test.pkl')
    grp = d.groupby(['subject','method'])
    m = grp.mean()
    gel = m.reset_index(level=['subject','method'])
    geu = m.reset_index(level=['subject','method'])

    tl, tu = scipy.stats.norm.ppf(1-confidence), scipy.stats.norm.ppf(confidence)
    tl, tu = scipy.stats.norm.interval(alpha=1-confidence, loc=grp.mean(), scale=grp.std())
    gel['ratio'] = tl
    geu['ratio'] = tu    
    
    
    #fig = px.scatter(m, error_y=e_upper, error_y_minus=e_lower, color='subject')
    fig = px.strip(grp.mean().reset_index(level=['subject','method']), x='method', y='ratio', color='subject')
    fig.add_traces( px.strip(gel, x='method', y='ratio', color='subject').data )
    fig.add_traces( px.strip(geu, x='method', y='ratio', color='subject').data )
    fig.add_hline(1.0 - float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio', range=[-0.1,1.1])
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test} | FWH = {fwh}')
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
    fig.update_yaxes(title='Positive Ratio', range=[0, 1.2])
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test}')
    if show:
        fig.show()
    fig.write_image(f'{test}_{confidence}.jpg', scale=2)


def plot_inter(directory, test, confidence, subjects, show=False):

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
    fig.add_hline(float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio', range=[0, 1.2])
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test} | {"".join(args.subjects)}')
    if show:
        fig.show()
    fig.write_image(
        f'{test}_{confidence}_{"".join(args.subjects)}.jpg', scale=2)


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--violin', action='store_true')
    parser.add_argument('--directory', default='mri_pickle')
    parser.add_argument('--test', choices=tests_name.keys(), required=True)
    parser.add_argument('--subjects', required=True, nargs='+')
    parser.add_argument('--confidence', required=True, type=float)
    parser.add_argument('--fwh', type=int, action='store', help='FWH')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--print-args', action='store_true',
                        help="Show passed args")
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    if args.print_args:
        print(args)

    if args.test == 'one':
        plot_one(args)

    elif args.test == 'inter':
        plot_inter(args)

    elif args.violin:
        plot_violin(args)

    else:
        plot_box(args)
