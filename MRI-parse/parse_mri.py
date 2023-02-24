import argparse
import glob
import os
import pickle
from statistics import NormalDist
from unicodedata import category

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import scipy.stats
import seaborn as sns
from plotly.subplots import make_subplots

pio.kaleido.scope.mathjax = None

tests_name = {
    'exclude': 'all-exclude',
    'include': 'all-include',
    'one': 'one',
    'inter': 'one'
}

methods_name = [
    'pce',
    'fdr_BH',
    'fdr_BY',
    'fwe_simes_hochberg',
    'fwe_holm_bonferroni',
    'fwe_holm_sidak',
    'fwe_sidak',
    'fwe_bonferroni'
]

ref_subjects = [
    'sub-adult15',
    'sub-adult16',
    'sub-xp201',
    'sub-xp207',
    'sub-1',
    'sub-36',
    'sub-CTS201',
    'sub-CTS210'
]


def open_file(filename):
    with open(filename, 'rb') as fi:
        return pickle.load(fi)
    return None


def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    return dist.mean - h, dist.mean + h

#   one_0.95_reference_rr_ds002338_sub-xp201_target_rs_ds002338_sub-xp201.pkl


def get_test(test, confidence,
             reference, reference_subject,
             target, target_subject,
             fwh):
    regexp = [test, confidence, 'reference', reference, reference_subject,
              'target', target, target_subject,
              'fwh', fwh]
    regexp = '_'.join(regexp)
    files = glob.glob(regexp)
    if len(files) == 0:
        return []
    [include] = files
    return include


def get_pce(args, df, alpha, alternative='two-sided'):
    '''
    Return tests that passes
    '''
    def ttest(sample, mean):
        return scipy.stats.ttest_1samp(sample, popmean=mean, alternative=alternative).pvalue

    df = df[df['method'] == 'pce']

    indexes = ['dataset', 'subject', 'confidence', 'fwh', 'sample_size']

    try:
        drop = ['k_fold', 'k_round', 'target', 'method']
        pvalues = df.groupby(indexes).agg(list).drop(drop, axis=1).apply(
            lambda t: ttest(t.fvr, 1 - t.name[2]), axis=1, result_type='expand')
    except KeyError:
        drop = ['kth_round', 'nb_round', 'target', 'method']
        pvalues = df.groupby(indexes).agg(list).drop(drop, axis=1).apply(
            lambda t: ttest(t.fvr, 1 - t.name[2]), axis=1, result_type='expand')
    return pvalues > alpha


def get_mct(args, df, alpha, alternative='two-sided', ratio=False):
    '''
    Return tests that passes
    '''

    names = ['dataset', 'subject', 'confidence',
             'fwh', 'sample_size', 'method']

    df = df[df['method'] != 'pce']
    df = df[df['method'] != 'fdr_TSBY']
    df = df[df['method'] != 'fdr_TSBH']
    df = df[df['method'] != 'fdr_BH']
    df = df[df['method'] != 'fwe_simes_hochberg']
    df = df[df['method'] != 'fwe_sidak']
    df = df[df['method'] != 'fwe_holm_sidak']
    df = df[df['method'] != 'fwe_holm_bonferroni']
    # df = df[df['method'] != 'fdr_BY']
    df = df[df['method'] != 'fwe_bonferroni']

    indexes = ['dataset', 'subject', 'confidence',
               'fwh', 'sample_size', 'method']

    def binom(fail, trials, alpha):
        return scipy.stats.binomtest(k=fail, n=trials, p=alpha, alternative=alternative).pvalue

    print('MCT')

    try:
        drop = ['k_fold', 'k_round']
        group = df.groupby(indexes).agg([np.sum, 'count']).drop(drop, axis=1)
    except KeyError:
        drop = ['kth_round', 'nb_round']
        group = df.groupby(indexes).agg([np.sum, 'count']).drop(drop, axis=1)

    if ratio:
        # print(group.apply(lambda t: t))
        ratio = group.apply(
            lambda t: t.fvr['sum'] / t.fvr['count'], axis=1, result_type='expand')
        # print(ratio)
        return ratio
    else:
        pvalues = group.apply(lambda t: binom(int(t.fvr['sum']),
                                              t.fvr['count'], alpha),
                              axis=1, result_type='expand')
        # print(pvalues)
        return pvalues > alpha


def plot_pce(pces):

    colors = ['crimson', 'forestgreen'] + (['orange'] if args.show_nan else [])

    subjects = pces[0].reset_index()['subject'].unique()
    cols = len(pces)
    rows = len(subjects)

    pce_fig = make_subplots(rows=rows, cols=cols,
                            column_titles=['RR', 'RS', 'RR+RS'],
                            row_titles=subjects.tolist(),
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='FWHM',
                            y_title='Confidence level',
                            vertical_spacing=0.02,
                            horizontal_spacing=0.01)

    for col, pce in enumerate(pces, start=1):

        pce = pce.reset_index()
        for row, subject in enumerate(subjects, start=1):

            for a in pce_fig['layout']['annotations']:
                a['textangle'] = 0

            pce_ = pce[pce['subject'] == subject]
            pce_2d = pce_.reset_index().pivot(
                index=['confidence'], columns=['fwh'], values=0)
            pce_2d_sorted = pce_2d.sort_index(
                axis=1).sort_index(axis=0, ascending=True)

            pce_x_labels = [t for t in pce_2d_sorted.sort_index(
                axis=1).columns.values]
            pce_y_labels = [t for t in pce_2d_sorted.index.values]

            p = pce_2d_sorted.replace({False: 0, True: 1, np.nan: 2})
            im = px.imshow(p, zmin=0, zmax=1,
                           color_continuous_scale=colors,
                           x=pce_x_labels, y=pce_y_labels,
                           origin='lower')

            pce_fig.add_trace(im.data[0], row=row, col=col)

    pce_fig.update_layout(coloraxis=dict(colorscale=colors))
    pce_fig.update_coloraxes(cmin=0, cmax=1)
    pce_fig.update_traces(showlegend=False)
    pce_fig.update_coloraxes(showscale=False)
    pce_fig.update_layout(margin=dict(t=25, b=60))
    pce_fig['layout']['annotations'][-1]['textangle'] = -90

    return pce_fig


def plot_mct(mcts, ratio=False):

    title = f'{args.title} ({args.meta_alpha})'
    subjects = mcts[0].reset_index()['subject'].unique()
    cols = len(mcts)
    rows = len(subjects)

    if ratio:
        colors = 'RdYlGn_r'
        zmin = 0
        zmax = 1
    else:
        colors = ['rgb(165,0,38)', 'forestgreen'] + \
            (['orange'] if args.show_nan else [])
        zmin = 0
        zmax = 2 if args.show_nan else 1

    mct_fig = make_subplots(rows=rows, cols=cols,
                            column_titles=['RR', 'RS', 'RR+RS'],
                            row_titles=subjects.tolist(),
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='FWHM',
                            y_title='Confidence level',
                            vertical_spacing=0.02,
                            horizontal_spacing=0.01)

    for col, mct in enumerate(mcts, start=1):

        mct = mct.reset_index()

        for a in mct_fig['layout']['annotations']:
            a['textangle'] = 0

        for row, subject in enumerate(subjects, start=1):
            print(col, row, subject)

            mct_ = mct[mct['subject'] == subject]
            mct_2d = mct_.pivot(
                index=['confidence', 'method'], columns=['fwh'], values=0)
            mct_2d_sorted = mct_2d.sort_index(
                axis=1).sort_index(axis=0, ascending=True)

            mct_x_labels = [t for t in mct_2d_sorted.sort_index(
                axis=1).columns.values]
            mct_y_labels = [
                str(t[0]) for t in mct_2d_sorted.sort_index(axis=1, ascending=True).index.values]

            if ratio:
                p = mct_2d_sorted
            else:
                p = mct_2d_sorted.replace({False: 0, True: 1, np.nan: 2})

            im = px.imshow(p,
                           color_continuous_scale=colors,
                           x=mct_x_labels, y=mct_y_labels,
                           zmin=zmin, zmax=zmax,
                           origin='upper')
            print(im)
            mct_fig.add_trace(im.data[0], row=row, col=col)

    mct_fig.update_layout(coloraxis=dict(colorscale=colors))
    mct_fig.update_layout(title='')
    if not args.ratio:
        mct_fig.update_traces(showlegend=False)
        mct_fig.update_coloraxes(showscale=False)
    else:
        # mct_fig.update_layout(margin=dict(r=225))
        mct_fig.update_layout(coloraxis_colorbar_x=1.05)
    mct_fig.update_layout(margin=dict(t=25))
    mct_fig['layout']['annotations'][-1]['textangle'] = -90

    return mct_fig


def plotly_backend(args, pces, mcts, show, no_pce, no_mct, ratio=False):
    pce_fig = plot_pce(pces)
    mct_fig = plot_mct(mcts, ratio)

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()

    ext = '_ratio' if args.ratio else ''

    pce_fig.write_image(f'{args.test}_pce{ext}.pdf', scale=5)
    mct_fig.write_image(f'{args.test}_mct{ext}.pdf', scale=5)


def seaborn_backend(pce, mct, show):

    sns.set_theme(style='dark', palette='pastel')

    mct_2d = mct.reset_index().pivot(
        index=['confidence', 'method'], columns=['fwh', 'subject'], values=0)
    mct_2d_sorted = mct_2d.sort_index(
        axis=1).sort_index(axis=0, ascending=False)

    mct_x_labels = [' '.join(map(str, t))
                    for t in mct_2d_sorted.sort_index(axis=1).columns.values]
    mct_y_labels = [' '.join(map(str, t))
                    for t in mct_2d_sorted.sort_index(axis=1).index.values]

    f, ax = plt.subplots()
    sns.heatmap(mct_2d_sorted.replace({False: 0, True: 1, np.nan: 2}), cmap=[
        'red', 'green', 'orange'], ax=ax, cbar_kws={'ticks': [0, 1, 2]},
        square=True, xticklabels=mct_x_labels, yticklabels=mct_y_labels)

    ax2 = ax.twiny()
    ax2.set_xticks(range(len(mct_x_labels)))
    ax2.set_xticklabels([int(float(t.split()[0]))
                        for t in mct_x_labels])
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.collections[0].colorbar.set_ticklabels(['Reject', 'Pass', 'NA'])

    if show:
        plt.show()


def get_optimum(df):
    df = df.reset_index()
    g = df.groupby(['confidence', 'fwh'])
    s = g.sum().drop('sample_size', axis=1)
    optimum = s.loc[s[0].max() == s[0]]
    indexes = optimum.index.values
    (alpha_star, fwh_star) = max(indexes, key=lambda t: (t[0], -t[1]))
    return (alpha_star, fwh_star)


def plot_exclude(args):
    pd.set_option('display.max_rows', 1500)

    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    ext = '_ratio' if args.ratio else ''

    dfs = []

    for reference in references:
        paths = glob.glob(f'{reference}/*.pkl')
        ldf = []
        for path in paths:
            with open(path, 'rb') as fib:
                pkl = pickle.load(fib)
                df = pd.DataFrame(pkl)
                df.insert(0, "prefix", reference)
                ldf.append(df)
        dfs.append(pd.concat(ldf))

    pce_tests, mct_tests = [], []
    for df in dfs:
        pce_tests.append(get_pce(args, df, alpha, alternative='greater'))
        mct_tests.append(
            get_mct(args, df, alpha, alternative='greater', ratio=args.ratio))

    for reference, pce_test, mct_test in zip(references, pce_tests, mct_tests):
        print('=' * 30)
        print(reference)
        (alpha_star, fwh_star) = get_optimum(pce_test)
        print(f'pce alpha*={alpha_star}, fwh*={fwh_star}')
        (alpha_star, fwh_star) = get_optimum(mct_test)
        print(f'mct alpha*={alpha_star}, fwh*={fwh_star}')
        name = reference.replace(os.path.sep, '_')
        pce_test.to_csv(f'{args.test}_{name}_pce{ext}.csv')
        mct_test.to_csv(f'{args.test}_{name}_mct{ext}.csv')

    plotly_backend(args,
                   pce_tests, mct_tests, show,
                   no_pce=args.no_pce,
                   no_mct=args.no_mct,
                   ratio=args.ratio)


def plot_one(args):
    plot_exclude(args)


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--directory', default='mri_pickle')
    parser.add_argument('--reference', required=True, nargs='+')
    parser.add_argument('--test', choices=tests_name.keys(), required=True)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--print-args', action='store_true',
                        help="Show passed args")
    parser.add_argument('--meta-alpha', default=0.05, type=float,
                        help='Alpha on permutation test')
    parser.add_argument('--title', default='')
    parser.add_argument('--no-pce', action='store_true',
                        help='Do not show PCE')
    parser.add_argument('--no-mct', action='store_true',
                        help='Do not show MCT')
    parser.add_argument('--show-nan', action='store_true', help='Show NaN')
    parser.add_argument('--ratio', action='store_true', help='Print ratio')
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    if args.print_args:
        print(args)

    elif args.test == 'exclude':
        plot_exclude(args)
    elif args.test == 'one':
        plot_one(args)
