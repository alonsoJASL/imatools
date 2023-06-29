import os
from imatools.common.ioutils import cout, fullfile
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_dict(mydic, plotname, out_dir, oname, ylims=[]):
    """
    Plot dictionary.
    plot_params can be:
        [ymin ymax], xlabel, ylabel
    """
    fig, ax = plt.subplots()
    try:
        func = getattr(ax, plotname)
        func(mydic.values())
        ax.set_xticklabels(mydic.keys())

        if ylims:
            ax.set_ylim(ylims[0], ylims[1])

        fig.suptitle(oname)
        fig.savefig(fullfile(out_dir, '{}_{}.pdf'.format(oname, plotname)))
        return fig, ax

    except AttributeError:
        cout("Plot function {} does not exist".format(plotname), 'ERROR')
        sys.exit(-1)

def extract_scar_stats_from_file(filename: str) : 
    """
    Extracts scar stats from prodStats.txt file.
    """
    fname = os.path.normpath(filename)
    print(fname)
    scar_stats = {}
    with open(fname, 'r') as f : 
        lines = f.readlines()
        method = lines[0]
        bp_mean = lines[1]
        bp_std = lines[2]

        # format is 'V=value, SCORE=score'
        scar_stats = { 'value_score' : [] }
        for line in lines[3:] :
            line = line.strip()
            if line == '' : 
                continue
            values, score = line.split(',')
            values = values.split('=')[1]
            score = score.split('=')[1]
            scar_stats['value_score'].append((float(values), float(score)))
        
        scar_stats['method'] = method
        scar_stats['bp_mean'] = float(bp_mean)
        scar_stats['bp_std'] = float(bp_std)

    return scar_stats


def append_scar_to_pandas_dataframe(df: pd.DataFrame, scar_stats: dict, case_id = '', roi_mode = '', roi_limits = '', thresh = '') : 
    """
    Append scar stats to pandas dataframe.
    """
    
    for tu in scar_stats['value_score'] :
        df = pd.concat([df, 
        pd.DataFrame({
            'case_id' : case_id,
            'roi_mode' : roi_mode,
            'threshold_method' : thresh,
            'roi_limits' : roi_limits,
            'bp_mean' : scar_stats['bp_mean'],
            'bp_std' : scar_stats['bp_std'],
            'values' : tu[0],
            'score' : tu[1]
            }, index=[0])], ignore_index=True)

    return df