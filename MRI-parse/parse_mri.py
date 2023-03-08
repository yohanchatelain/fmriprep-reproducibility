import itertools
import sys
import argparse
import glob
import os
import pickle
from statistics import NormalDist
from unicodedata import category

import matplotlib.pyplot as plt
import numpy as np
import polars as pd
import plotly.express as px
import plotly.io as pio
import scipy.stats
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

pandas_library = pd.__name__


def filter(table, column_name, column_value):
    if pandas_library == 'pandas':
        return table[table[column_name] == column_value]
    elif pandas_library == 'polars':
        return table.filter((pd.col(column_name) == column_value))
    else:
        raise Exception(f'Unkown table library {pd.__name_}')


def insert(table, column_name, column_value):
    if pandas_library == 'pandas':
        table.insert(0, column_name, column_value)
        return table
    elif pandas_library == 'polars':
        return table.with_columns(
            pd.Series(name=column_name, values=column_value))
    else:
        raise Exception(f'Unkown table library {pd.__name_}')


def add_prefix(table, prefix):
    if pandas_library == 'pandas':
        return insert(table, 'prefix', prefix)
    elif pandas_library == 'polars':
        return insert(table, 'prefix', [prefix])
    else:
        raise Exception(f'Unkown table library {pd.__name_}')


def drop_column(table, columns_name):
    if pandas_library == 'pandas':
        return table.drop(columns_name, axis=1)
    elif pandas_library == 'polars':
        return table.drop(columns_name)
    else:
        raise Exception(f'Unkown table library {pd.__name_}')


string_methods = {'pandas': {'endswith': 'endswith', 'contains': '__contains__'},
                  'polars': {'endswith': 'ends_with', 'contains': 'contains'}}


def filter_string(table, column_name, regexp, method):
    if pandas_library == 'pandas':
        if method == 'contains':
            return filter_string_contains(table, column_name, regexp)
    elif pandas_library == 'polars':
        if method == 'contains':
            return filter_string_contains(table, column_name, regexp)
    else:
        raise Exception(f'Unkown table library {pd.__name_}')


def filter_string_contains(table, column_name, regexp):
    if pandas_library == 'pandas':
        return table[regexp in table[column_name]]
    elif pandas_library == 'polars':
        return table.filter(pd.col(column_name).str.contains(regexp))
    else:
        raise Exception(f'Unkown table library {pd.__name_}')


def open_file(filename):
    with open(filename, 'rb') as fi:
        return pickle.load(fi)
    return None


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


def get_pce_exclude(args, df, alpha, alternative='two-sided', ratio=False):
    '''
    Return tests that passes
    '''
    def ttest(sample, mean):
        return scipy.stats.ttest_1samp(sample, popmean=mean, alternative=alternative).pvalue

    df = df[df['method'] == 'pce']

    indexes = ['dataset', 'subject', 'confidence', 'fwh', 'sample_size']

    drop = ['kth_round', 'nb_round', 'target', 'method']

    df.insert(0, 'alpha', value=1 - df['confidence'])
    df.insert(0, 'fvr', value=df['reject'] / df['tests'])

    if args.ratio:
        ratio = df.groupby(indexes).agg(list).drop(drop, axis=1).apply(
            lambda t: np.mean(t.fvr), axis=1, result_type='expand')
        return ratio
    else:
        pvalues = df.groupby(indexes).agg(list).drop(drop, axis=1).apply(
            lambda t: ttest(t.fvr, t.alpha[0]), axis=1, result_type='expand')
        return pvalues > alpha


def get_pce_one(args, df, alpha, ratio=False):

    df = filter(df, 'method', 'pce')

    indexes = ['dataset', 'subject', 'confidence',
               'fwh',  'target']
    drop = ['kth_round', 'nb_round', 'method',
            'sample_size', 'reject', 'tests']

    df = insert(df, 'success', df['reject'] /
                df['tests'] <= (1 - df['confidence']))

    df = drop_column(df.groupby(indexes).apply(lambda t: t), drop)

    return filter_string_contains(df, 'target', '.1/fmriprep')


def get_mct_one(args, df, alpha, ratio=False):

    df = df[df['method'] != 'pce']
    df = df[df['method'] != 'fdr_TSBY']
    df = df[df['method'] != 'fdr_TSBH']
    df = df[df['method'] != 'fdr_BH']
    df = df[df['method'] != 'fwe_simes_hochberg']
    df = df[df['method'] != 'fwe_sidak']
    df = df[df['method'] != 'fwe_holm_sidak']
    df = df[df['method'] != 'fwe_holm_bonferroni']
    df = df[df['method'] != 'fdr_BY']

    indexes = ['dataset', 'subject', 'confidence',
               'fwh', 'sample_size', 'target']
    drop = ['kth_round', 'nb_round', 'method']

    df['success'] = df['reject'] / df['tests'] <= (1 - df['confidence'])
    df = df.groupby(indexes).apply(lambda t: t).drop(drop, axis=1)
    return df[df['target'].endswith('.1')]


def get_pce_deviation(args, df):
    df = df[df['method'] == 'pce']

    indexes = ['prefix',  'dataset', 'subject',
               'confidence',
               'fwh',
               'sample_size']
    drop = ['kth_round', 'nb_round', 'target']
    print(df)
    df['alpha'] = 1 - df['confidence']
    df['positive'] = df['reject'] / df['tests']
    x = df.drop(drop, axis=1).groupby(indexes).agg(
        [np.mean, np.var, list]).apply(lambda t: t)
    print(x)
    x['neff'] = (x['alpha']['mean'] *
                 (1-x['alpha']['mean'])) / x['positive']['var']
    y = x[x['positive']['mean'] != 0]
    print('y', y)
    z = y.apply(lambda t: scipy.stats.norm.sf(
        t['positive']['list'], loc=t['alpha']['mean'], scale=1/t['neff']), axis=1)
    # print(z)

    print('z', z.groupby(indexes).apply(
        lambda t: (np.mean(t[-1]), np.std(t[-1]))))
    sys.exit(1)


def get_mct_exclude(args, df, alpha, alternative='two-sided', ratio=False):
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
    df = df[df['method'] != 'fdr_BY']
    # df = df[df['method'] != 'fwe_bonferroni']

    indexes = ['dataset', 'subject', 'confidence',
               'fwh', 'sample_size', 'method']

    def binom(fail, trials, alpha):
        return scipy.stats.binomtest(k=fail, n=trials, p=alpha, alternative=alternative).pvalue

    print('MCT')

    df['fail'] = df['reject'] > 0
    print(df)
    try:
        drop = ['k_fold', 'k_round']
        group = df.groupby(indexes).agg([np.sum, 'count']).drop(drop, axis=1)
    except KeyError:
        drop = ['kth_round', 'nb_round']
        group = df.groupby(indexes).agg([np.sum, 'count']).drop(drop, axis=1)

    print(group)
    if ratio:
        # print(group.apply(lambda t: t))
        ratio = group.apply(
            lambda t: t.fail['sum'] / t.fail['count'], axis=1, result_type='expand')
        # print(ratio)
        return ratio
    else:
        pvalues = group.apply(lambda t: binom(int(t.fail['sum']),
                                              t.fail['count'], alpha),
                              axis=1, result_type='expand')
        # print(pvalues)
        return pvalues > alpha


def plot_pce_exclude(pces, ratio=False):

    if ratio:
        colors = 'RdYlGn_r'
        zmin = 0
        zmax = 1
    else:
        colors = ['rgb(165,0,38)', 'forestgreen'] + \
            (['orange'] if args.show_nan else [])
        zmin = 0
        zmax = 2 if args.show_nan else 1

    subjects = pces[0].reset_index()['subject'].unique()
    cols = len(pces)
    rows = len(subjects)

    pce_fig = make_subplots(rows=rows, cols=cols,
                            column_titles=['RR', 'RS', 'RR+RS'],
                            row_titles=subjects.tolist(),
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='FWHM (mm)',
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
            im = px.imshow(p, zmin=zmin, zmax=zmax,
                           color_continuous_scale=colors,
                           x=pce_x_labels, y=pce_y_labels,
                           origin='lower')

            pce_fig.add_trace(im.data[0], row=row, col=col)

    pce_fig.update_layout(coloraxis=dict(colorscale=colors))
    pce_fig.update_coloraxes(cmin=0, cmax=1)
    pce_fig.update_layout(margin=dict(t=25, b=60))

    if not args.ratio:
        pce_fig.update_traces(showlegend=False)
        pce_fig.update_coloraxes(showscale=False)
    else:
        pce_fig.update_layout(coloraxis_colorbar_x=1.05)

    pce_fig['layout']['annotations'][-1]['textangle'] = -90

    return pce_fig


def plot_pce_one(pces, ratio=False):
    colors = ['rgb(165,0,38)', 'forestgreen'] + \
        (['orange'] if args.show_nan else [])

    pd.Config().set_fmt_str_lengths(150)

    for pce in pces:
        if pandas_library == 'polars':
            pce = pce.with_columns([pce['success'].cast(pd.Int32)])
            pce = pce.sort("confidence", "fwh", "subject",
                           descending=[True, False, False])
        if pandas_library == 'pandas':
            pce = pce.reset_index()

        fwhs = map(float, pce['fwh'].unique())
        confidences = map(float, pce['confidence'].unique())

        for fwh, confidence in itertools.product(fwhs, confidences):

            print(pce)

            pce_filtered = pce.filter(
                (pd.col('fwh') == fwh) & (pd.col('confidence') == confidence)
            )

            print('filtered', pce_filtered)

            pce2d = pce_filtered.pivot(index=['confidence'], columns=[
                'fwh'], values='success')

            print('pce2d', pce2d)

            if pandas_library == 'pandas':
                pce2d_sorted = pce2d.sort_index(
                    axis=1).sort_index(axis=0, ascending=True)
            else:
                pce2d_sorted = pce2d
            print(pce2d_sorted)
            if pandas_library == 'pandas':
                p = pce2d_sorted.replace({False: 0, True: 1})
            else:
                p = pce2d_sorted
            fig = px.imshow(p, zmin=0, zmax=1,
                            color_continuous_scale=colors, origin='lower')
            fig.show()


def plot_mct_one(mcts, ratio=False):
    colors = ['rgb(165,0,38)', 'forestgreen'] + \
        (['orange'] if args.show_nan else [])
    for mct in mcts:
        mct2d = mct.reset_index().pivot(
            index=['confidence'], columns=['fwh'], values=0)
        mct2d_sorted = mct2d.sort_index(
            axis=1).sort_index(axis=0, ascending=True)
        p = mct2d_sorted.replace({False: 0, True: 1})
        fig = px.imshow(p, zmin=0, zmax=1,
                        color_continuous_scale=colors, origin='lower')
        fig.show()


def plot_mct_exclude(mcts, ratio=False):

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
                            x_title='FWHM (mm)',
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
            # print(im)
            mct_fig.add_trace(im.data[0], row=row, col=col)

    mct_fig.update_layout(coloraxis=dict(colorscale=colors))
    mct_fig.update_layout(title='')
    if not args.ratio:
        mct_fig.update_traces(showlegend=False)
        mct_fig.update_coloraxes(showscale=False)
    else:
        mct_fig.update_layout(coloraxis_colorbar_x=1.05)
    mct_fig.update_layout(margin=dict(t=25))
    mct_fig['layout']['annotations'][-1]['textangle'] = -90

    return mct_fig


def plotly_backend_exclude(args, pces, mcts, show, no_pce, no_mct, ratio=False):
    if not no_pce:
        pce_fig = plot_pce_exclude(pces, ratio)
    if not no_mct:
        mct_fig = plot_mct_exclude(mcts, ratio)

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()

    ext = '_ratio' if args.ratio else ''

    if not no_pce:
        pce_fig.write_image(f'{args.test}_pce{ext}.pdf', scale=5)
    if not no_mct:
        mct_fig.write_image(f'{args.test}_mct{ext}.pdf', scale=5)


def plotly_backend_one(args, pces, mcts, show, no_pce, no_mct, ratio=False):
    if not no_pce:
        pce_fig = plot_pce_one(pces, ratio)
    if not no_mct:
        mct_fig = plot_mct_one(mcts, ratio)

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()

    ext = '_ratio' if args.ratio else ''

    if not no_pce:
        pce_fig.write_image(f'{args.test}_pce{ext}.pdf', scale=5)
    if not no_mct:
        mct_fig.write_image(f'{args.test}_mct{ext}.pdf', scale=5)


def get_optimum(df):
    print(df)
    if pandas_library == 'pandas':
        df = df.reset_index()
    g = df.groupby(['confidence', 'fwh'])
    s = drop_column(g.sum(), 'sample_size')

    optimum = s.loc[s[0].max() == s[0]]
    indexes = optimum.index.values
    (alpha_star, fwh_star) = max(indexes, key=lambda t: (t[0], -t[1]))
    return (alpha_star, fwh_star)


def parse_dataframe(dfs, get_test, **kwds):
    return list(map(lambda df: get_test(args, df, **kwds), dfs))


def get_optimum_test(references, pce_tests, ext):
    for reference, pce_test in zip(references, pce_tests):
        print('=' * 30)
        print(reference)
        (alpha_star, fwh_star) = get_optimum(pce_test)
        print(f'pce alpha*={alpha_star}, fwh*={fwh_star}')
        name = reference.replace(os.path.sep, '_')
        pce_test.to_csv(f'{args.test}_{name}_pce{ext}.csv')


def get_references(references):
    dfs = []
    for reference in references:
        paths = glob.glob(f'{reference}/*.pkl')
        ldf = []
        for path in paths:
            with open(path, 'rb') as fib:
                pkl = pickle.load(fib)
                df = pd.DataFrame(pkl)
                ldf.append(add_prefix(df, reference))
        dfs.append(pd.concat(ldf))
    return dfs


def plot_exclude(args):
    pd.set_option('display.max_rows', 1500)

    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    ext = '_ratio' if args.ratio else ''

    dfs = get_references(references)

    if not args.no_pce:
        pce_tests = parse_dataframe(
            dfs,  get_pce_exclude, alpha=alpha, alternative='greater', ratio=args.ratio)
        get_optimum_test(references, pce_tests, ext)
    else:
        pce_tests = []

    if not args.no_mct:
        mct_tests = parse_dataframe(
            dfs,  get_mct_exclude, alpha=alpha, alternative='greater', ratio=args.ratio)
        get_optimum_test(references, mct_tests, ext)
    else:
        mct_tests = []

    plotly_backend_exclude(args,
                           pce_tests, mct_tests, show,
                           no_pce=args.no_pce,
                           no_mct=args.no_mct,
                           ratio=args.ratio)


def plot_one(args):
    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    ext = '_ratio' if args.ratio else ''

    dfs = get_references(references)
    if not args.no_pce:
        pce_tests = parse_dataframe(
            dfs,  get_pce_one, alpha=alpha, ratio=args.ratio)
        # get_optimum_test(references, pce_tests, ext)
    else:
        pce_tests = []

    if not args.no_mct:
        mct_tests = parse_dataframe(
            dfs,  get_mct_one, alpha=alpha,  ratio=args.ratio)
        # get_optimum_test(references, mct_tests, ext)
    else:
        mct_tests = []

    plotly_backend_one(args,
                       pce_tests, mct_tests, show,
                       no_pce=args.no_pce,
                       no_mct=args.no_mct,
                       ratio=args.ratio)


def plot_deviation_exclude(args):
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('max_colwidth', -1)

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
        pce_tests.append(
            get_pce_deviation(args, df))

    for reference, pce_test in zip(references, pce_tests):
        print('=' * 30)
        print(reference)
        (alpha_star, fwh_star) = get_optimum(pce_test)
        print(f'pce alpha*={alpha_star}, fwh*={fwh_star}')
        name = reference.replace(os.path.sep, '_')
        pce_test.to_csv(f'{args.test}_{name}_pce{ext}.csv')

    plotly_backend(args,
                   pce_tests, mct_tests, show,
                   no_pce=args.no_pce,
                   no_mct=args.no_mct,
                   ratio=args.ratio)


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
    parser.add_argument('--deviation', action='store_true',
                        help='Compute deviation statistics')
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    if args.print_args:
        print(args)

    elif args.test == 'exclude':
        if args.deviation:
            plot_deviation_exclude(args)
        else:
            plot_exclude(args)
    elif args.test == 'one':
        plot_one(args)
