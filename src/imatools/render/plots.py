"""
Generic dict-plotting and scar-stats-to-dataframe helpers.

Stateless data-in/figure-or-dataframe-out functions, plus one small file
reader (``extract_scar_stats_from_file``) that was already file-path-based
in ``plotutils.py`` (kept verbatim — behaviour-preserving relocation only,
no new I/O added).
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dict(mydic, plotname, out_dir: Path, oname, ylims=[]):
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
        fig.savefig(out_dir / f"{oname}_{plotname}.pdf")

        return fig, ax

    except AttributeError:
        print(f"[ERROR] Plot function {plotname} does not exist")
        # BUG (pre-existing in master common/plotutils.py): `sys` was never imported —
        # this branch raises NameError instead of exiting. Preserved verbatim (Cat-B
        # latent bug, dead unless `plotname` doesn't resolve to an Axes method); do not
        # fix under a pure-relocation task.
        sys.exit(-1)  # noqa: F821


def extract_scar_stats_from_file(filename: str):
    """
    Extracts scar stats from prodStats.txt file.
    """
    fname = os.path.normpath(filename)
    print(fname)
    scar_stats = {}
    with open(fname, "r") as f:
        lines = f.readlines()
        method = lines[0]
        bp_mean = lines[1]
        bp_std = lines[2]

        # format is 'V=value, SCORE=score'
        scar_stats = {"value_score": []}
        for line in lines[3:]:
            line = line.strip()
            if line == "":
                continue
            values, score = line.split(",")
            values = values.split("=")[1]
            score = score.split("=")[1]
            scar_stats["value_score"].append((float(values), float(score)))

        scar_stats["method"] = method
        scar_stats["bp_mean"] = float(bp_mean)
        scar_stats["bp_std"] = float(bp_std)

    return scar_stats


def append_scar_to_pandas_dataframe(df: pd.DataFrame, scar_stats: dict, case_info: dict):
    # case_id = '', roi_mode = '', roi_limits = '', thresh = '') :
    """
    Append scar stats to pandas dataframe.
    """

    for tu in scar_stats["value_score"]:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "case_id": case_info["case_id"],
                        "nav": case_info["nav"],
                        "roi_mode": case_info["roi_mode"],
                        "roi_limits": case_info["roi_limits"],
                        "threshold_method": case_info["thresh"],
                        "bp_mean": scar_stats["bp_mean"],
                        "bp_std": scar_stats["bp_std"],
                        "values": tu[0],
                        "score": tu[1],
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    return df
