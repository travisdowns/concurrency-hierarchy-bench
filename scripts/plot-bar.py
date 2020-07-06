#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
import os

# for arguments that should be comma-separate lists, we use splitlsit as the type
splitlist = lambda x: x.split(',')

p = argparse.ArgumentParser(usage='plot output from ./bench')

p.add_argument('--procs', help='Number of processors used (CPUS in data.sh)', default=4, type=int)

# input and output file configuration
p.add_argument('input', help='CSV file to plot (or stdin)', type=argparse.FileType('r'), default=[ sys.stdin ])
p.add_argument('--out', help='output directory')
p.add_argument('--table-out', help='output directory for HTML tables')
p.add_argument('--show', help='also show output interactively (default if --out is not specified)', nargs='?', const='all')

# input parsing configuration
p.add_argument('--sep', help='separator character (or regex) for input', default=',')

# chart type

# column selection and configuration

# chart labels and text
p.add_argument('--clabels', help="Comma separated list of column names used as label for data series (default: column header)",
    type=splitlist)
p.add_argument('--xlabel', help='Set x axis label', default='Active Cores')
p.add_argument('--ylabel', help='Set y axis label', default='Nanoseconds per increment')
p.add_argument('--ylabel2', help='Set the secondary y axis label')

# legend
p.add_argument('--legend-loc', help='Set the legend location explicitly', type=str)

# data manipulation
p.add_argument('--jitter', help='Apply horizontal (x-axis) jitter of the given relative amount (default 0.1)',
    nargs='?', type=float, const=0.1)
p.add_argument('--group', help='Group data by the first column, with new min/median/max columns with one row per group')

# axis and line/point configuration
p.add_argument('--ylim', help='Set the y axis limits explicitly (e.g., to cross at zero)', type=float, nargs='+')
p.add_argument('--xrotate', help='rotate the xlablels by this amount', default=0)
p.add_argument('--tick-interval', help='use the given x-axis tick spacing (in x axis units)', type=int)
p.add_argument('--alpha', help='use the given alpha for marker/line', type=float)
p.add_argument('--linewidth', help='use the given line width', type=float)
p.add_argument('--tight', help='use tight_layout for less space around chart', action='store_true', default=True)

# debugging
p.add_argument('--verbose', '-v', help='enable verbose logging', action='store_true')
cargs = p.parse_args()

vprint = print if cargs.verbose else lambda *a: None
vprint("cargs = ", cargs)

plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'large'

# fix various random seeds so we get reproducible plots
# fix the mpl seed used to generate SVG IDs
mpl.rcParams['svg.hashsalt'] = 'foobar'

# numpy random seeds (used by e.g., jitter function below)
np.random.seed(123)


kwargs = {}

if (cargs.alpha):
    kwargs['alpha'] = cargs.alpha

# these are args that are basically just passed directly through to the plot command
# and generally correspond to matplotlib plot argumnets.
passthru_args  = ['markersize', 'marker', 'color']
passthru_args2 = ['markersize2', 'marker2', 'color2']
argsdict = vars(cargs)


# populate the per-series arguments, based on the series index
def populate_args(idx, base, secondary = False):
    assert idx > 0
    idx = idx - 1 # because the columns are effectively 1-based (col 0 are the x values)
    arglist = passthru_args2 if secondary else passthru_args
    kwargs = base.copy()
    for arg in arglist:
        argval = argsdict[arg]
        argname = arg[:-1] if secondary and arg.endswith('2') else arg
        if (argval):
            kwargs[argname] = argval[idx % len(argval)]
            vprint("set {} for {} col {} to {} (list: {})".format(argname, "secondary" if secondary else "primary", idx, kwargs[argname], argval))
        else:
            vprint("not set {} for col {}".format(arg, idx))
    return kwargs


fullargs = {}
vprint("kwargs: {}".format(fullargs))

df = pd.read_csv(cargs.input, sep=cargs.sep, index_col=[0, 1, 2])
df.sort_index(level=0, inplace=True)
vprint("----- from file -------\n", df.head(), "\n---------------------")


def make_plot(filename, title, cols, minthreads=1, maxthreads=cargs.procs, overlay=[], metrics='Nanos/Op'):
    subf = df[metrics].copy()
    vprint("----- after metric slice ------\n", subf.head(), "\n---------------------")
    subf = subf.unstack()
    vprint("----- after reshape ------\n", subf.head(), "\n---------------------")
    subf = subf[cols]
    vprint("----- after col selection ------\n", subf.head(), "\n---------------------")
    # subf = subf.unstack().droplevel(axis='columns', level=0)
    vprint("-----  columns   ------\n", subf.columns, "\n---------------------")

    iv = subf.index.get_level_values('Cores')
    vprint('iv:', iv)
    subf = subf.loc[(iv >= minthreads) & (iv <= maxthreads), :]
    vprint("----- after core filter ------\n", subf.head(), "\n---------------------")

    gb = subf.groupby(by=['Cores']);
    median = gb.median()
    vprint("----- after groupby ------\n", median.head(n=20), "\n---------------------")

    p10 = median - gb.quantile(0.10)
    p90 = gb.quantile(0.90) - median
    vprint("----- p10 ------\n", p10.head(), "\n---------------------")
    vprint("----- p90 ------\n", p90.head(), "\n---------------------")


    p10v = p10[cols[0]].values;
    vprint('p10v: ', p10v)
    vprint('p10v shape:', np.shape(p10v))

    # https://stackoverflow.com/a/37139647
    # y error specification should have shape
    # ( number of columns, 2, number of rows )
    # or equivalently
    # ( number of bars within a group, 2, number of groups )
    err = []
    for col in median:
        err.append([p10[col].values, p90[col].values])
    # vprint('err:', err)
    vprint('err shape:', np.shape(err))

    ax = median.plot.bar(title=title, figsize=(9,6), rot=0, yerr=err, **fullargs)

    vprint('>>>>', ax.containers)

    # # add overlay to the bars
    if overlay:
        idx = 0
        for bars in ax.containers:
            # print('bars: ', bars)
            # print('type: ', type(bars))
            if isinstance(bars, mpl.container.BarContainer):
                for rect in bars:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., height/2,
                            overlay[idx],
                            ha='center', va='bottom', rotation=0, fontsize=16, weight='bold')
                    idx = idx + 1

    if cargs.xrotate:
        plt.xticks(rotation=cargs.xrotate)

    if cargs.ylabel:
        ax.set_ylabel(cargs.ylabel)

    if cargs.ylim:
        if len(cargs.ylim) == 1:
            ax.set_ylim(cargs.ylim[0])
        elif len(cargs.ylim) == 2:
            ax.set_ylim(cargs.ylim[0], cargs.ylim[1])
        else:
            sys.exit('provide one or two args to --ylim')

    # this needs to go after the ax2 handling, or else secondary axis x label will override
    if cargs.xlabel:
        ax.set_xlabel(cargs.xlabel)

    legargs = {}
    if cargs.legend_loc:
        legargs['loc'] = cargs.legend_loc

    if cargs.tight:
        plt.tight_layout()

    if cargs.out:
        outpath = os.path.join(cargs.out, filename + '.svg')
        vprint("Saving figure to ", outpath, "...")
        plt.savefig(outpath)

    if cargs.show and (cargs.show == 'all' or filename in cargs.show.split(',')):
        vprint("Showing interactive plot...")
        plt.show()

    plt.close()

    if cargs.table_out:
        # this line moves the index name to be the first column name instead
        # subf = subf.rename_axis(index=None, columns=subf.index.name)
        header = "---\nlayout: default\n---\n\n"
        tpath = os.path.join(cargs.table_out, filename + '.html')
        with open(tpath, 'w') as f:
            f.write(header + subf.to_html())
        vprint('saved html table to', tpath)


columns = []

def ac(*args):
    columns.extend([*args])
    return columns


make_plot('mutex',      'Increment Cost: std::mutex', ac('mutex add'), minthreads=2)
make_plot('atomic-inc', 'Increment Cost: Atomic Increments', ac('atomic add', 'cas add'), minthreads=2)
make_plot('atomic-inc1','Increment Cost: Atomic Increments', ac(), maxthreads=1)
make_plot('fair-yield', 'Increment Cost: Yielding Ticket', ac('ticket yield'), minthreads=2)
make_plot('more-fair',  'Increment Cost: More Fair Locks', ac('ticket blocking', 'queued fifo'), minthreads=2)
make_plot('ts-4',       'Increment Cost: Ticket Spin', ac('ticket spin'), minthreads=2)
make_plot('ts-6',       'Increment Cost: Ticket Spin (Oversubscribed)', ac(), minthreads=2, maxthreads=(cargs.procs + 2))

make_plot('single',     'Increment Cost: Single Threaded', ac(), maxthreads=1, overlay=[2, 1, 1, 1, 3, 4, 1])

columns = ['atomic add', 'cas multi']
make_plot('cas-multi', 'Increment Cost: Contention Adaptive Multi-Counter', ac())
make_plot('tls',       'Increment Cost: Thread Local Storage', ac('tls add'))