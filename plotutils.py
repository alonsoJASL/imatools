from ioutils import cout, fullfile
import pandas as pd
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
