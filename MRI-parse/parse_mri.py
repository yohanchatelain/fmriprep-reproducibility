from plotly.subplots import make_subplots
import seaborn as sns
import scipy.stats
import plotly.graph_objects as go
import glob
import plotly.express as px
import os
import pickle
import sys
import argparse
import pandas as pd
from unicodedata import category
from statistics import NormalDist
import json
import numpy as np
import matplotlib.pyplot as plt

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
    'sub-CTS201',
    'sub-CTS210'
]


def open_file(filename):
    with open(filename, 'rb') as fi:
        return pickle.load(fi)
    return None


# def get_test(directory, test, confidence, reference, reference_subject, target, target_subject, verbose=False):
#     test_name = tests_name[test]
#     regexp = f'{directory}{os.sep}{test_name}*{confidence}_reference_{reference}_*_{reference_subject}_target_{target}_*_{target_subject}*fwh_{fwh}.pkl'
#     files = glob.glob(regexp)
#     print(regexp)
#     print(files)
#     if files == []:
#         return []
#     [exclude] = files
#     return open_file(exclude)


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


# def test_pce(df, alpha, alternative='two-sided'):
#     '''
#     df: Datafram with all information
#     '''

#     # FrozenList(['dataset', 'subject', 'confidence', 'sample_size', 'fwh'])
#     index_names = ['dataset', 'subject', 'confidence', 'sample_size', 'fwh']

#     def ttest(fvr, confidence):
#         return scipy.stats.ttest_1samp(fvr, popmean=1-confidence, alternative=alternative).pvalue

#     df = df[df['method'] == 'pce'].drop(['target', 'k_fold', 'k_round', 'method'], axis=1).groupby(
#         ['dataset', 'subject', 'confidence', 'sample_size', 'fwh']).agg(list)

#     assert(index_names == df.index.names)

#     pvalues = df.apply(lambda t:  ttest(
#         fvr=t.fvr, confidence=t.name[2]), axis=1)

#     return pvalues > alpha


def get_pce(df, alpha, alternative='two-sided'):
    '''
    Return tests that passes
    '''

    df = df[df['method'] == 'pce']

    indexes = ['dataset', 'subject', 'confidence', 'fwh', 'sample_size']
    drop = ['k_fold', 'k_round', 'target', 'method']

    def ttest(sample, mean):
        return scipy.stats.ttest_1samp(sample, popmean=mean, alternative=alternative).pvalue

    pvalues = df.groupby(indexes).agg(list).drop(drop, axis=1).apply(
        lambda t: ttest(t.fvr, 1-t.name[2]), axis=1, result_type='expand')

    return pvalues > alpha


def get_mct(df, alpha, alternative='two-sided'):
    '''
    Return tests that passes
    '''

    names = ['dataset', 'subject', 'confidence',
             'fwh', 'sample_size', 'method']

    df = df[df['method'] != 'pce']
    df = df[df['method'] != 'fdr_TSBY']
    df = df[df['method'] != 'fdr_TSBH']
    df = df[df['method'] != 'fdr_BH']

    indexes = ['dataset', 'subject', 'confidence',
               'fwh', 'sample_size', 'method']
    drop = ['k_fold', 'k_round']

    def binom(fail, trials, alpha):
        return scipy.stats.binomtest(k=fail, n=trials, p=alpha, alternative=alternative).pvalue

    group = df.groupby(indexes).agg([np.sum, 'count']).drop(drop, axis=1)

    keys = dict(map(lambda t: t[::-1], enumerate(group.index.names)))
    pvalues = group.apply(lambda t: binom(
        int(t.fvr['sum']), t.fvr['count'], 1 - t.name[keys['confidence']]), axis=1, result_type='expand')

    return pvalues > alpha


def plotly_backend(pce, mct, show, no_pce, no_mct):

    title = f'{args.title} ({args.meta_alpha})'

    pce_2d = pce.reset_index().pivot(
        index=['confidence'], columns=['fwh', 'subject'], values=0)
    pce_2d_sorted = pce_2d.sort_index(
        axis=1).sort_index(axis=0, ascending=True)

    colors = ['red', 'green'] + (['orange'] if args.show_nan else [])

    pce_x_labels = [' '.join(map(str, t))
                    for t in pce_2d_sorted.sort_index(axis=1).columns.values]
    pce_y_labels = [t for t in pce_2d_sorted.index.values]

    pce_fig = px.imshow(pce_2d_sorted.replace({False: 0, True: 1, np.nan: 2}), color_continuous_scale=colors,
                        x=pce_x_labels, y=pce_y_labels, origin='lower')

    pce_fig.update_layout(title=title)
    pce_fig.update_layout(xaxis2=dict(title='FWHM', domain=[0.25, 0.75]))

    mct_2d = mct.reset_index().pivot(
        index=['confidence', 'method'], columns=['fwh', 'subject'], values=0)
    mct_2d_sorted = mct_2d.sort_index(
        axis=1).sort_index(axis=0, ascending=False)

    pd.set_option('display.max_rows', 500)

    print(mct_2d_sorted.mean())
    print(mct_2d_sorted.mean(axis=1))

    mct_x_labels = [' '.join(map(str, t))
                    for t in mct_2d_sorted.sort_index(axis=1).columns.values]
    mct_y_labels = [' '.join(map(str, t))
                    for t in mct_2d_sorted.sort_index(axis=1).index.values]

    mct_fig = px.imshow(mct_2d_sorted.replace(
        {False: 0, True: 1, np.nan: 2}), color_continuous_scale=colors, x=mct_x_labels, y=mct_y_labels)
    mct_fig.update_layout(title=title)

    fhw_sep = []
    confidence_sep = []
    x_labels = mct_fig.data[0]['x']
    y_labels = mct_fig.data[0]['y']

    x_new_labels = []
    y_new_labels = []

    fwh_before = 0
    for x in x_labels:
        fwh, subject = x.split()
        fwh = int(float(fwh))
        new_label = f'{fwh} {subject}'
        x_new_labels.append(new_label)
        if fwh_before != fwh:
            if len(x_new_labels) > 2:
                fhw_sep.append(x_new_labels[-2])
            fwh_before = fwh

    confidence_before = 0.005
    for y in y_labels:
        confidence, method = y.split()
        # confidence = 1 - float(confidence)
        new_label = f'{confidence} {method}'
        y_new_labels.append(new_label)
        if confidence_before != confidence:
            if len(y_new_labels) > 2:
                confidence_sep.append(y_new_labels[-2])
            confidence_before = confidence

    for sep in fhw_sep:
        mct_fig.add_vline(x=sep, opacity=0.2)

    for sep in confidence_sep:
        mct_fig.add_hline(y=sep, opacity=0.2)

    mct_fig.data[0]['x'] = x_new_labels
    mct_fig.data[0]['y'] = y_new_labels

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()


def plotly_backend_split(pce, mct, show, no_pce, no_mct):
    # pce = pce.reset_index()

    # title = f'{args.title} ({args.meta_alpha})'

    # pce_fig = make_subplots(
    #     rows=4, cols=2, subplot_titles=pce['subject'].unique())
    # subjects = pce['subject'].unique().reshape((4, 2))
    # for (row, col), subject in np.ndenumerate(subjects):

    #     pce_ = pce[pce['subject'] == subject]
    #     pce_2d = pce_.reset_index().pivot(
    #         index=['confidence'], columns=['fwh'], values=0)
    #     pce_2d_sorted = pce_2d.sort_index(
    #         axis=1).sort_index(axis=0, ascending=True)

    #     pce_x_labels = [t for t in pce_2d_sorted.sort_index(
    #         axis=1).columns.values]
    #     pce_y_labels = [t for t in pce_2d_sorted.index.values]
    #     colors = ['red', 'green'] + (['orange'] if args.show_nan else [])

    #     p = pce_2d_sorted.replace({False: 0, True: 1, np.nan: 2})
    #     im = px.imshow(p,
    #                    color_continuous_scale=colors,
    #                    x=pce_x_labels, y=pce_y_labels,
    #                    origin='lower')
    #     pce_fig.add_trace(im.data[0], row=row + 1, col=col + 1)

    # pce_fig.update_layout(coloraxis=dict(colorscale=colors))
    # pce_fig.update_layout(title=title)
    # pce_fig.show()

    title = f'{args.title} ({args.meta_alpha})'

    mct = mct.reset_index()
    mct_fig = make_subplots(
        rows=4, cols=2, subplot_titles=mct['subject'].unique())
    subjects = mct['subject'].unique().reshape((4, 2))
    for (row, col), subject in np.ndenumerate(subjects):

        mct_ = mct[mct['subject'] == subject]
        mct_2d = mct_.pivot(
            index=['confidence', 'method'], columns=['fwh'], values=0)
        mct_2d_sorted = mct_2d.sort_index(
            axis=1).sort_index(axis=0, ascending=True)

        mct_x_labels = [t for t in mct_2d_sorted.sort_index(
            axis=1).columns.values]
        mct_y_labels = [' '.join(map(str, t))
                        for t in mct_2d_sorted.sort_index(axis=1).index.values]

        colors = ['red', 'green'] + (['orange'] if args.show_nan else [])

        p = mct_2d_sorted.replace({False: 0, True: 1, np.nan: 2})
        im = px.imshow(p,
                       color_continuous_scale=colors,
                       x=mct_x_labels, y=mct_y_labels,
                       origin='lower')
        mct_fig.add_trace(im.data[0], row=row + 1, col=col + 1)

    mct_fig.update_layout(coloraxis=dict(colorscale=colors))
    mct_fig.update_layout(title=title)
    mct_fig.show()
    return

    mct_2d = mct.reset_index().pivot(
        index=['confidence', 'method'], columns=['fwh', 'subject'], values=0)
    mct_2d_sorted = mct_2d.sort_index(
        axis=1).sort_index(axis=0, ascending=False)

    pd.set_option('display.max_rows', 500)
    print(mct_2d_sorted.mean())
    print(mct_2d_sorted.mean(axis=1))

    mct_x_labels = [' '.join(map(str, t))
                    for t in mct_2d_sorted.sort_index(axis=1).columns.values]
    mct_y_labels = [' '.join(map(str, t))
                    for t in mct_2d_sorted.sort_index(axis=1).index.values]

    mct_fig = px.imshow(mct_2d_sorted.replace(
        {False: 0, True: 1, np.nan: 2}), color_continuous_scale=colors, x=mct_x_labels, y=mct_y_labels)
    mct_fig.update_layout(title=title)

    fhw_sep = []
    confidence_sep = []
    x_labels = mct_fig.data[0]['x']
    y_labels = mct_fig.data[0]['y']

    x_new_labels = []
    y_new_labels = []

    fwh_before = 0
    for x in x_labels:
        fwh, subject = x.split()
        fwh = int(float(fwh))
        new_label = f'{fwh} {subject}'
        x_new_labels.append(new_label)
        if fwh_before != fwh:
            if len(x_new_labels) > 2:
                fhw_sep.append(x_new_labels[-2])
            fwh_before = fwh

    confidence_before = 0.005
    for y in y_labels:
        confidence, method = y.split()
        # confidence = 1 - float(confidence)
        new_label = f'{confidence} {method}'
        y_new_labels.append(new_label)
        if confidence_before != confidence:
            if len(y_new_labels) > 2:
                confidence_sep.append(y_new_labels[-2])
            confidence_before = confidence

    for sep in fhw_sep:
        mct_fig.add_vline(x=sep, opacity=0.2)

    for sep in confidence_sep:
        mct_fig.add_hline(y=sep, opacity=0.2)

    mct_fig.data[0]['x'] = x_new_labels
    mct_fig.data[0]['y'] = y_new_labels

    if show:
        if not no_pce:
            pce_fig.show()
        if not no_mct:
            mct_fig.show()


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


def plot_exclude(args):

    reference = args.reference
    show = args.show
    alpha = args.meta_alpha

    dfs = []

    paths = glob.glob(f'{reference}/*.pkl')
    for path in paths:
        with open(path, 'rb') as fib:
            pkl = pickle.load(fib)
            dfs.append(pd.DataFrame(pkl))

    dfs = pd.concat(dfs)

    pce_tests = get_pce(dfs, alpha, alternative='two-sided')
    mct_tests = get_mct(dfs, alpha, alternative='greater')

    if args.split:
        plotly_backend_split(pce_tests, mct_tests, show,
                             no_pce=args.no_pce, no_mct=args.no_mct)
    else:
        plotly_backend(pce_tests, mct_tests, show,
                       no_pce=args.no_pce, no_mct=args.no_mct)


def plot_test(args):

    test = args.test
    confidence = args.confidence
    reference = args.reference
    reference_inputs = args.reference_inputs
    target = args.target
    target_inputs = args.target_inputs
    fwh = args.fwh
    show = args.show

    dfs = []
    with open(reference_inputs, 'r') as fi:
        ref_json = json.load(fi)
        ds = [(dataset, subject) for dataset in ref_json.keys()
              for subject in ref_json[dataset].keys()]
        for dataset, subject in ds:
            pickle_path = '_'.join([test, str(confidence),
                                   'reference', reference, dataset, subject,
                                    'target', target, dataset, subject,
                                    'fwh', str(fwh)])
            path = os.path.join(args.reference, f'{pickle_path}.pkl')
            with open(path, 'rb') as fib:
                pkl = pickle.load(fib)
                dfs.append(pd.DataFrame(pkl))

    dfs = pd.concat(dfs)

    fig = go.Figure()

    pce = dfs[dfs['method'] == 'pce']
    pce_strip = px.strip(pce, x='method', y='fvr', color='subject')
    pce_mean = pce.groupby(['dataset', 'subject']).agg(
        [np.mean, np.std])['fvr']['mean']
    pce_std = pce.groupby(['dataset', 'subject']).agg(
        [np.mean, np.std])['fvr']['std']

    pce_mean_frame = pce_mean.to_frame()
    pce_mean_frame.index = [i[1] for i in pce_mean_frame.index.to_flat_index()]

    pce_std_frame = pce_std.to_frame()

    pce_stat = pd.concat([pce_mean_frame, pce_std_frame], axis=1)

    stat_fig = px.scatter(pce_stat, x=pce_stat.index,
                          y='mean', color=pce_stat.index, error_y='std')

    mct = dfs[dfs['method'] != 'pce']
    mct.groupby(['dataset', 'subject', 'sample_size', 'confidence']).sum()

    mcts = mct.groupby(
        ['dataset', 'subject', 'sample_size', 'confidence', 'method']).sum()

    ci = mcts.reset_index().apply(lambda t: (t.subject, t.method) +
                                  tuple(scipy.stats.binomtest(k=int(t.fvr), n=t.sample_size, p=1-t.confidence).proportion_ci()) +
                                  (t.fvr/t.sample_size,),
                                  axis=1, result_type='expand')

    ci.columns = ['subject', 'method', 'low', 'high', 'mean']

    # Reshape to have a matrix : z.to_frame().reset_index().pivot(columns=['fwh','subject'], index=['confidence','method'], values=0)
    # mis.columns = list(map(lambda t:(int(t[0]),t[1]), mis.columns))
    # mis.columns = pd.MultiIndex.from_tuples(mis.columns, names=['fwh','subject'])


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

    d = pd.DataFrame.from_dict({'subject': __subject,
                                'ratio': __fvr,
                               'method': __method})
    d.to_pickle('test.pkl')
    grp = d.groupby(['subject', 'method'])
    m = grp.mean()
    gel = m.reset_index(level=['subject', 'method'])
    geu = m.reset_index(level=['subject', 'method'])

    tl, tu = scipy.stats.norm.ppf(
        1-confidence), scipy.stats.norm.ppf(confidence)
    tl, tu = scipy.stats.norm.interval(
        alpha=0.05, loc=grp.mean(), scale=grp.std())
    gel['ratio'] = tl
    geu['ratio'] = tu

    # fig = px.scatter(m, error_y=e_upper, error_y_minus=e_lower, color='subject')
    fig = px.strip(grp.mean().reset_index(
        level=['subject', 'method']), x='method', y='ratio', color='subject')
    fig.add_traces(px.strip(gel, x='method', y='ratio', color='subject').data)
    fig.add_traces(px.strip(geu, x='method', y='ratio', color='subject').data)
    fig.add_hline(1.0 - float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio', range=[-0.1, 1.1])
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test} | FWH = {fwh}')
    if show:
        fig.show()
    fig.write_image(f'{test}_{confidence}.jpg')


def plot_one(args):

    test = args.test
    confidence = args.confidence
    reference = args.reference
    reference_inputs = args.reference_inputs
    target = args.target
    target_inputs = args.target_inputs
    fwh = args.fwh
    show = args.show

    dl = []

    __subject = []
    __fvr = []
    __method = []

    for reference_input in reference_inputs:
        for target_input in target_inputs:
            values = get_test(test, confidence, reference,
                              reference_input, target, target_inputs, fwh)
            if values == []:
                continue
            methods = dict(zip(methods_name, [[] for _ in methods_name]))
            for value in values:
                if value['method'] in methods:
                    __subject.append(reference_input)
                    __fvr.append(value['fvr'])
                    __method.append(value['method'])

    d = pd.DataFrame.from_dict({'subject': __subject,
                                'ratio': __fvr,
                               'method': __method})
    d.to_pickle('test.pkl')
    grp = d.groupby(['subject', 'method'])
    m = grp.mean()
    gel = m.reset_index(level=['subject', 'method'])
    geu = m.reset_index(level=['subject', 'method'])

    tl, tu = scipy.stats.norm.ppf(
        1-confidence), scipy.stats.norm.ppf(confidence)
    tl, tu = scipy.stats.norm.interval(
        alpha=0.05, loc=grp.mean(), scale=grp.std())
    gel['ratio'] = tl
    geu['ratio'] = tu

    # fig = px.scatter(m, error_y=e_upper, error_y_minus=e_lower, color='subject')
    fig = px.strip(grp.mean().reset_index(
        level=['subject', 'method']), x='method', y='ratio', color='subject')
    fig.add_traces(px.strip(gel, x='method', y='ratio', color='subject').data)
    fig.add_traces(px.strip(geu, x='method', y='ratio', color='subject').data)
    fig.add_hline(1.0 - float(confidence))
    fig.update_xaxes(title='Methods')
    fig.update_yaxes(title='Positive Ratio', range=[-0.1, 1.1])
    fig.update_layout(
        title=f'Confidence level = {confidence} | Test = {test} | FWH = {fwh}')
    if show:
        fig.show()
    fig.write_image(f'{test}_{confidence}.jpg')


def plot_inter(args):

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
    fig = px.box(d, color='subject', box=True)
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
    parser.add_argument('--target')
    parser.add_argument('--target-input')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference-input', required=False,
                        help='JSON file with dataset to plot')
    parser.add_argument('--test', choices=tests_name.keys(), required=True)
    parser.add_argument('--subjects', required=False, nargs='+')
    parser.add_argument('--confidence', required=False, type=float)
    parser.add_argument('--fwh', type=int, action='store', help='FWH')
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
    parser.add_argument('--split', action='store_true', help='split image')
    return parser.parse_args()


if '__main__' == __name__:
    args = parse_args()
    if args.print_args:
        print(args)

    if args.test == 'one':
        plot_one(args)

    elif args.test == 'inter':
        plot_inter(args)

    elif args.test == 'exclude':
        plot_exclude(args)

    elif args.violin:
        plot_violin(args)

    else:
        plot_box(args)
