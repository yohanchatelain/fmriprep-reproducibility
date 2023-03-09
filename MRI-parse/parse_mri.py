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


def get_pce_exclude(args, df, alpha,
                    alternative='two-sided',
                    ratio=False,
                    high_confidence=False):
    '''
    Return tests that passes
    '''
    def ttest(series_of_struct):
        _values = [
            scipy.stats.ttest_1samp(next_struct['fvr'],
                                    popmean=next_struct['alpha'],
                                    alternative=alternative).pvalue
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    def ttest_ci(series_of_struct):
        _values = [
            tuple(scipy.stats.ttest_1samp(next_struct['fvr'],
                                          popmean=next_struct['alpha'],
                                          alternative=alternative).confidence_interval(confidence_level=1-alpha))
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    df = df.filter((pd.col('method') == 'pce'))

    if high_confidence:
        df = df.filter(pd.col('confidence') > 0.99)
    else:
        df = df.filter(pd.col('confidence') <= 0.99)

    indexes = ['reference_dataset', 'reference_subject', 'reference_template',
               'target_dataset', 'target_subject', 'target_template',
               'confidence', 'fwhm', 'alpha',
               'mask']

    df = df.with_columns(
        (
            (1 - pd.col('confidence')).alias('alpha'),
            (pd.col('reject') / pd.col('tests')).alias('fvr')
        )
    )

    df = df.groupby(indexes).agg(
        [
            pd.col('fvr'),
            pd.col('fvr').mean().alias('ratio')
        ]
    )

    df = df.with_columns(
        (
            (pd.struct(["fvr", "alpha"]).map(ttest)).alias('pvalue')
        )
    )

    df = df.with_columns(
        (pd.col('pvalue') >= alpha).alias('success')
    )

    return df


def get_pce_one(args, df, alpha,
                alternative='two-sided',
                ratio=False,
                high_confidence=False):
    '''
    Return tests that passes
    '''
    # def ttest(series_of_struct):
    #     _values = [
    #         scipy.stats.ttest_1samp(next_struct['fvr'],
    #                                 popmean=next_struct['alpha'],
    #                                 alternative=alternative).pvalue
    #         for next_struct in series_of_struct
    #     ]
    #     return pd.Series(values=_values)

    # def ttest_ci(series_of_struct):
    #     _values = [
    #         tuple(scipy.stats.ttest_1samp(next_struct['fvr'],
    #                                       popmean=next_struct['alpha'],
    #                                       alternative=alternative).confidence_interval(confidence_level=1-alpha))
    #         for next_struct in series_of_struct
    #     ]
    #     return pd.Series(values=_values)

    df = df.filter((pd.col('method') == 'pce'))

    if high_confidence:
        df = df.filter(pd.col('confidence') > 0.99)
    else:
        df = df.filter(pd.col('confidence') <= 0.99)

    # indexes = ['reference_dataset', 'reference_subject', 'reference_template',
    #            'target_dataset', 'target_subject', 'target_template',
    #            'confidence', 'fwhm', 'alpha',
    #            'mask']

    df = df.with_columns(
        (
            (1 - pd.col('confidence')).alias('alpha'),
            (pd.col('reject') / pd.col('tests')).alias('fvr')
        )
    )

    # df = df.groupby(indexes).agg(
    #     [
    #         pd.col('fvr'),
    #         pd.col('fvr').mean().alias('ratio')
    #     ]
    # )

    # df = df.with_columns(
    #     (
    #         (pd.struct(["fvr", "alpha"]).map(ttest)).alias('pvalue')
    #     )
    # )

    df = df.with_columns(
        (pd.col('fvr') <= pd.col('alpha')).alias('success')
    )

    return df


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


def get_mct_exclude(args, df, alpha,
                    alternative='two-sided',
                    ratio=False,
                    method='fwe_bonferroni',
                    high_confidence=False):
    '''
    Return tests that passes
    '''
    def binom(series_of_struct):
        _values = [
            scipy.stats.binomtest(k=next_struct['fails'],
                                  n=next_struct['trials'],
                                  p=next_struct['alpha'],
                                  alternative=alternative).pvalue
            for next_struct in series_of_struct
        ]
        return pd.Series(values=_values)

    indexes = ['reference_dataset', 'reference_subject', 'reference_template',
               'target_dataset', 'target_subject', 'target_template',
               'confidence', 'fwhm', 'alpha',
               'mask']

    df = df.filter((pd.col('method') == method))

    if high_confidence:
        df = df.filter(pd.col('confidence') > 0.99)
    else:
        df = df.filter(pd.col('confidence') <= 0.99)

    df = df.with_columns(
        (
            (pd.col('reject') > 0).alias('fail'),
            (1 - pd.col('confidence')).alias('alpha')
        )
    )

    df = df.groupby(indexes).agg(
        [
            (pd.col('fail').sum()).alias('fails'),
            (pd.col('fail').count()).alias('trials'),
            (pd.col('fail').mean()).alias('ratio')
        ]
    )

    df = df.with_columns(
        (pd.struct(["fails", "trials", "alpha"]).map(binom).alias('pvalue'))
    )

    df = df.with_columns(
        (pd.col('pvalue') >= alpha).alias('success')
    )

    return df


def get_mct_one(args, df, alpha,
                alternative='two-sided',
                ratio=False,
                method='fwe_bonferroni',
                high_confidence=False):
    '''
    Return tests that passes
    '''
    # def binom(series_of_struct):
    #     _values = [
    #         scipy.stats.binomtest(k=next_struct['fails'],
    #                               n=next_struct['trials'],
    #                               p=next_struct['alpha'],
    #                               alternative=alternative).pvalue
    #         for next_struct in series_of_struct
    #     ]
    #     return pd.Series(values=_values)

    # indexes = ['reference_dataset', 'reference_subject', 'reference_template',
    #            'target_dataset', 'target_subject', 'target_template',
    #            'confidence', 'fwhm', 'alpha',
    #            'mask']

    df = df.filter((pd.col('method') == method))

    if high_confidence:
        df = df.filter(pd.col('confidence') > 0.99)
    else:
        df = df.filter(pd.col('confidence') <= 0.99)

    df = df.with_columns(
        (
            (pd.col('reject') > 0).alias('fail'),
            (1 - pd.col('confidence')).alias('alpha')
        )
    )

    # df = df.groupby(indexes).agg(
    #     [
    #         (pd.col('fail').sum()).alias('fails'),
    #         (pd.col('fail').count()).alias('trials'),
    #         (pd.col('fail').mean()).alias('ratio')
    #     ]
    # )

    # df = df.with_columns(
    #     (pd.struct(["fails", "trials", "alpha"]).map(binom).alias('pvalue'))
    # )

    df = df.with_columns(
        (pd.col('reject') <= 0).alias('success')
    )

    return df


def plot_pce_exclude(pces, ratio=False, verbose=False):

    if ratio:
        colors = 'RdYlGn_r'
        zmin = 0
        zmax = 1
    else:
        colors = ['rgb(165,0,38)', 'forestgreen'] + \
            (['orange'] if args.show_nan else [])
        zmin = 0
        zmax = 2 if args.show_nan else 1

    subjects = pces[0].collect().select(
        pd.col('reference_subject')).unique().sort(by=['reference_subject']).to_dict(as_series=False)['reference_subject']
    cols = len(pces)
    rows = len(subjects)

    pce_fig = make_subplots(rows=rows, cols=cols,
                            column_titles=['RR', 'RS', 'RR+RS'],
                            row_titles=subjects,
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='FWHM (mm)',
                            y_title='Confidence level',
                            vertical_spacing=0.02,
                            horizontal_spacing=0.01)

    for col, pce in enumerate(pces, start=1):

        for row, subject in enumerate(subjects, start=1):

            for a in pce_fig['layout']['annotations']:
                a['textangle'] = 0

            pce_subject = pce.filter(pd.col('reference_subject') == subject).sort(
                by=['confidence', 'fwhm'], descending=[False, False]).collect()

            if ratio:
                pivot = pce_subject.pivot(index=['confidence'], columns=[
                    'fwhm'], values='ratio')
            else:
                pivot = pce_subject.pivot(index=['confidence'], columns=[
                    'fwhm'], values='success')

            confidences = pivot['confidence'].to_numpy()
            fwhms = pivot.columns[1:]
            z = pivot.to_numpy()[..., 1:]

            if verbose:
                print(subject)
                print('x', confidences.shape)
                print(confidences)
                print('y', len(fwhms))
                print(fwhms)
                print('z', z.shape)
                print(pivot)

            im = px.imshow(z,
                           x=[str(f) for f in fwhms],
                           y=[str(f) for f in confidences],
                           zmin=zmin, zmax=zmax,
                           color_continuous_scale=colors,
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


def plot_pce_one(pces, ratio=False, verbose=False):

    if ratio:
        colors = 'RdYlGn_r'
        zmin = 0
        zmax = 1
    else:
        colors = ['rgb(165,0,38)', 'forestgreen'] + \
            (['orange'] if args.show_nan else [])
        zmin = 0
        zmax = 2 if args.show_nan else 1

    subjects = pces[0].collect().select(
        pd.col('reference_subject')).unique().sort(by=['reference_subject']).to_dict(as_series=False)['reference_subject']

    pce_figs = []

    for pce in pces:

        pce = pce.collect()

        confidences = pce['confidence'].unique().sort(
            descending=True).to_numpy()
        fwhms = pce['fwhm'].unique().sort().to_numpy()

        print(confidences)
        print(fwhms)

        rows = confidences.size
        cols = fwhms.size

        pce_fig = make_subplots(rows=rows, cols=cols,
                                column_titles=[str(f) for f in fwhms],
                                row_titles=[str(c) for c in confidences],
                                shared_xaxes=True,
                                shared_yaxes=True,
                                x_title='FWHM (mm)',
                                y_title='Confidence level',
                                vertical_spacing=0,
                                horizontal_spacing=0)

        for row, confidence in enumerate(confidences, start=1):

            for a in pce_fig['layout']['annotations']:
                a['textangle'] = 0

            for col, fwhm in enumerate(fwhms, start=1):

                cell = pce.filter((pd.col('confidence') == confidence) & (
                    pd.col('fwhm') == fwhm)).sort(by=['confidence',
                                                      'fwhm',
                                                      'reference_subject',
                                                      'target_subject'])
                pivot = cell.pivot(index=['reference_subject'], columns=[
                    'target_subject'], values='success')

                x = pivot['reference_subject'].to_numpy()
                y = pivot.columns[1:]
                z = pivot.to_numpy()[..., 1:]

                if verbose:
                    print('='*30)
                    print(f'Confidence: {confidence}, FWHM: {fwhm}')
                    print('x', x.shape)
                    print(x)
                    print('y', len(y))
                    print(y)
                    print('z', z.shape)
                    print(pivot)
                    print(z)

                im = px.imshow(z,
                               x=[str(i) for i in x],
                               y=[str(i) for i in y],
                               zmin=zmin, zmax=zmax,
                               color_continuous_scale=colors,
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

        pce_figs.append(pce_fig)

    return pce_fig


def plot_mct_one(mcts, ratio=False, verbose=False):

    title = f'{args.title} ({args.meta_alpha})'
    subjects = mcts[0].collect().select(
        pd.col('reference_subject')).unique().sort(by=['reference_subject']).to_dict(as_series=False)['reference_subject']
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
                            row_titles=subjects,
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='FWHM (mm)',
                            y_title='Confidence level',
                            vertical_spacing=0.02,
                            horizontal_spacing=0.01)

    for col, mct in enumerate(mcts, start=1):

        for a in mct_fig['layout']['annotations']:
            a['textangle'] = 0

        for row, subject in enumerate(subjects, start=1):

            mct_subject = mct.filter(pd.col('reference_subject') == subject).sort(
                by=['confidence', 'fwhm'], descending=[False, False]).collect()

            if ratio:
                pivot = mct_subject.pivot(index=['confidence'], columns=[
                    'fwhm'], values='ratio')
            else:
                pivot = mct_subject.pivot(index=['confidence'], columns=[
                    'fwhm'], values='success')

            confidences = pivot['confidence'].to_numpy()
            fwhms = pivot.columns[1:]
            z = pivot.to_numpy()[..., 1:]

            if verbose:
                print(subject)
                print('x', confidences.shape)
                print(confidences)
                print('y', len(fwhms))
                print(fwhms)
                print('z', z.shape)
                print(pivot)

            im = px.imshow(z,
                           x=[str(f) for f in fwhms],
                           y=[str(f) for f in confidences],
                           zmin=zmin, zmax=zmax,
                           color_continuous_scale=colors,
                           origin='lower')
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


def plot_mct_exclude(mcts, ratio=False, verbose=False):

    title = f'{args.title} ({args.meta_alpha})'
    subjects = mcts[0].collect().select(
        pd.col('reference_subject')).unique().sort(by=['reference_subject']).to_dict(as_series=False)['reference_subject']
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
                            row_titles=subjects,
                            shared_xaxes=True,
                            shared_yaxes=True,
                            x_title='FWHM (mm)',
                            y_title='Confidence level',
                            vertical_spacing=0.02,
                            horizontal_spacing=0.01)

    for col, mct in enumerate(mcts, start=1):

        for a in mct_fig['layout']['annotations']:
            a['textangle'] = 0

        for row, subject in enumerate(subjects, start=1):

            mct_subject = mct.filter(pd.col('reference_subject') == subject).sort(
                by=['confidence', 'fwhm'], descending=[False, False]).collect()

            if ratio:
                pivot = mct_subject.pivot(index=['confidence'], columns=[
                    'fwhm'], values='ratio')
            else:
                pivot = mct_subject.pivot(index=['confidence'], columns=[
                    'fwhm'], values='success')

            confidences = pivot['confidence'].to_numpy()
            fwhms = pivot.columns[1:]
            z = pivot.to_numpy()[..., 1:]

            if verbose:
                print(subject)
                print('x', confidences.shape)
                print(confidences)
                print('y', len(fwhms))
                print(fwhms)
                print('z', z.shape)
                print(pivot)

            im = px.imshow(z,
                           x=[str(f) for f in fwhms],
                           y=[str(f) for f in confidences],
                           zmin=zmin, zmax=zmax,
                           color_continuous_scale=colors,
                           origin='lower')
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
        pce_fig = plot_pce_exclude(pces, ratio, args.verbose)
    if not no_mct:
        mct_fig = plot_mct_exclude(mcts, ratio, args.verbose)

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
        pce_fig = plot_pce_one(pces, ratio, args.verbose)
    if not no_mct:
        mct_fig = plot_mct_one(mcts, ratio, args.verbose)

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

    df = df.collect()
    confidences = df['confidence'].unique().sort()
    print('Local optimum per alpha threshold')
    for confidence in confidences:
        dfc = df.filter(pd.col('confidence') == confidence)
        argmax = dfc.select(pd.col('success').arg_max())
        argmax_df = (dfc.filter(pd.col('success') == argmax).sort(by=['confidence', 'fwhm']).select(
            [pd.col("alpha"), pd.col("fwhm")]))
        if argmax_df.height != 0:
            (alpha_star, fwhm_star) = argmax_df.to_numpy().min(axis=0)
            print(f' * {alpha_star:.6f}, {fwhm_star}')

    return (alpha_star, fwhm_star)


def parse_dataframe(dfs, get_test, **kwds):
    return list(map(lambda df: get_test(args, df, **kwds), dfs))


def get_optimum_test(references, tests, ext):
    for reference, test in zip(references, tests):
        print('=' * 30)
        print(reference)
        (alpha_star, fwh_star) = get_optimum(test)
        print(f'pce alpha*={alpha_star:.6f}, fwh*={fwh_star}')
        name = reference.replace(os.path.sep, '_')
        test.collect().to_pandas().to_csv(
            f'{args.test}_{name}_pce{ext}.csv')


def get_reference(reference):
    paths = glob.glob(f'{reference}/*.pkl')
    ldf = []
    for path in tqdm.tqdm(paths):
        with open(path, 'rb') as fib:
            pkl = pickle.load(fib)
            df = pd.DataFrame(pkl).lazy()
            ldf.append(df)
    return pd.concat(ldf)


def memoize(arg, fun):
    _raw_hash = hashlib.md5(arg.encode('utf-8')).hexdigest()
    _mem_file = f'{_raw_hash}.pkl'
    if os.path.exists(_mem_file):
        with open(_mem_file, 'rb') as fi:
            return pickle.load(fi)
    else:
        res = fun(arg)
        with open(_mem_file, 'wb') as fo:
            pickle.dump(res, fo)
        return res


def get_references(references):
    dfs = []
    for reference in references:
        # ldf = memoize(reference, get_reference)
        ldf = get_reference(reference)
        ldf = ldf.with_columns(
            (pd.Series(name='prefix', values=[reference]))
        )
        dfs.append(ldf)
    return dfs


def plot_exclude(args):
    # pd.Config().set_tbl_rows(1500)

    references = args.reference
    show = args.show
    alpha = args.meta_alpha

    ext = '_ratio' if args.ratio else ''

    dfs = get_references(references)

    if not args.no_pce:
        pce_tests = parse_dataframe(dfs,  get_pce_exclude,
                                    alpha=alpha,
                                    alternative='greater',
                                    ratio=args.ratio,
                                    high_confidence=args.high_confidence)
        get_optimum_test(references, pce_tests, ext)
    else:
        pce_tests = []

    if not args.no_mct:
        mct_tests = parse_dataframe(dfs,  get_mct_exclude,
                                    alpha=alpha,
                                    alternative='greater',
                                    ratio=args.ratio,
                                    method=args.mct_method,
                                    high_confidence=args.high_confidence)
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

    dfs = get_references(references)
    if not args.no_pce:
        pce_tests = parse_dataframe(dfs,  get_pce_one,
                                    alpha=alpha,
                                    alternative='greater',
                                    ratio=args.ratio,
                                    high_confidence=args.high_confidence
                                    )
    else:
        pce_tests = []

    if not args.no_mct:
        mct_tests = parse_dataframe(dfs,  get_mct_one,
                                    alpha=alpha,
                                    alternative='greater',
                                    ratio=args.ratio,
                                    method=args.mct_method,
                                    high_confidence=args.high_confidence
                                    )
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
    parser.add_argument('--mct-method', required=True)
    parser.add_argument('--high-confidence', action='store_true',
                        help='show confidence level 0.999 0.9999 0.99999 0.999999')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
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
