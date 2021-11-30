# plots.py
"""Generate all Figures used in the paper."""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from simulation_results import SimulationResults


_STYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-.")


def init_plt_settings(figsize=(12,3), fontsize="xx-large"):
    """Initialize custom matplotlib settings for paper Figures."""
    plt.rc("figure", figsize=figsize, dpi=300)
    plt.rc("axes", titlesize=fontsize, labelsize=fontsize, linewidth=.5)
    plt.rc("legend", fontsize=fontsize, edgecolor="none", frameon=False)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")


# Routines for each figure ====================================================

def convergence_singleparam():
    """Figure 1: Convergence in time for single-parameter estimation
    with various choices for α.

    Requires the folder data/convergence_singleparam/.
    """
    sr = SimulationResults("data/convergence_singleparam")

    # Plot results.
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,7))
    for index, ax in zip([0, 1, 2, 4], axes.flat):
        result = sr.results[index]
        result.plot(ax=ax)
        if index == 4:
            ax.set_title("No relaxation")
        else:
            ax.set_title(fr"$\alpha = {result.params['alpha']}$")

    # Format axes.
    for ax in axes.flat:
        ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
        ax.set_yticks([1e-14,
                       1e-12, 1e-11, 1e-10,
                       1e-8, 1e-7, 1e-6,
                       1e-4, 1e-3, 1e-2,
                       1e0, 1e1],
                      minor=True)
        ax.set_yticklabels([], minor=True)
        ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")
        ax.set_ylim([1e-15, 1e2])
    for ax in axes[-1,:]:
        ax.set_xlabel(r"Time $t$")
    for ax in axes[:,0]:
        ax.set_ylabel("Absolute error")

    for ax in axes.flat:
        for line, i in zip(ax.lines, [1, 6, 9]):
            line.set_color(f"C{i-1:d}")
            line.set_linewidth(1)
            line.set_linestyle(_STYLES[i-1])

    # Legend below the plots.
    fig.subplots_adjust(bottom=.2, wspace=.05)
    labels = [
        r"$|\lambda - \widehat{\lambda}(t)|$",
        r"$||I_h(u(\cdot,t)) - I_h(v(\cdot,t))||$",
        r"$||u(\cdot,t) - v(\cdot,t)||$"
    ]
    leg = axes[0,0].legend(labels, loc="lower center", ncol=3,
                           bbox_to_anchor=(.5,0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(4)

    fig.savefig("figures/convergence_singleparam.pdf",
                dpi=300, bbox_inches="tight")


def mu_alpha_convergence():
    """Figure 2: Convergence rate as functions of α and µ.

    Requires the folder data/mu_alpha_convergence.
    """
    results = SimulationResults("data/mu_alpha_convergence")
    summary = results.get_summary()
    # For convergence rate wrt α use the default value for µ = 1.8/δt.
    alpha_convergence = summary.iloc[:7]
    # For convergence wrt µ use the default value for α = 1.
    mu_convergence = summary.iloc[7:]

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,3))
    axes[0].axvline(mu_convergence["alpha"].iloc[0], color="C1", lw=.5)
    axes[0].semilogx(alpha_convergence["alpha"],
                     alpha_convergence["convergence_rate"],
                     "C0.-", lw=1, ms=8, mew=0)
    axes[0].text(mu_convergence["alpha"].iloc[0]*1.1, .25, r"$\alpha = 1$",
                 fontsize="large", color="C1", va="center", ha="left")
    # axes[0].axhline(alpha_convergence["convergence_rate"].iloc[2],
    #                 color="k", lw=.5)

    axes[1].axvline(alpha_convergence["mu"].iloc[0], color="C0", lw=.5)
    axes[1].semilogx(mu_convergence["mu"],
                     mu_convergence["convergence_rate"],
                     "C1.-", lw=1, ms=8, mew=0)
    axes[1].text(alpha_convergence["mu"].iloc[0]*.9, .25,
                 r"$\mu = 1.8/\delta t$",
                 fontsize="large", color="C0", va="center", ha="right")
    # axes[1].axhline(mu_convergence["convergence_rate"].iloc[-1],
    #                 color="k", lw=.5)

    axes[0].set_title(r"Fixed $\mu = 1.8/\delta t$")
    axes[0].set_xlabel(r"$\alpha$")
    axes[1].set_title(r"Fixed $\alpha = 1$")
    axes[1].set_xlabel(r"$\mu$")
    axes[0].set_ylabel(r"Convergence rate $\beta$")
    for ax in axes:
        ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")

    fig.subplots_adjust(wspace=.05)
    fig.savefig("figures/mu_alpha_convergence.pdf",
                dpi=300, bbox_inches="tight")


def finitedifference_order():
    """Figure 3: Convergence against time step for various FD schemes.

    Requires the folder data/finitedifference_order/.
    """
    sr = SimulationResults("data/finitedifference_order")

    # Extract and plot results.
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    df = sr.get_summary()
    data = df.groupby(["dt","order"])["lambda2_error"].sum().unstack()
    dt = data.index.values
    logdt = np.log10(dt)
    print("Estimated Orders of Accuracy:")
    for order, mark in zip(data.columns, "osd"):
        pts = data[order].values
        logdata = np.log10(pts[order-1:])
        estimated_order = stats.linregress(logdt[order-1:], logdata).slope
        print(order, estimated_order)
        ax.loglog(dt, pts, ls='-', lw=1, marker=mark, ms=8, mew=0)

    # Annotate each line (no legend).
    ax.text(9.5e-4, 1.5e-4, "first-order FD", color="C0", fontsize="large",
            ha="right", va="center")
    ax.text(9.5e-4, 1.1e-8, "second-order FD", color="C1", fontsize="large",
            ha="right", va="center")
    ax.text(1.1e-3, 6e-12, "third-order FD", color="C2", fontsize="large",
            ha="left", va="center")

    # Set labels and titles.
    ax.set_xlabel(r"Time step $\delta t$")
    ax.set_ylabel(r"$\displaystyle\int_{t_f-1}^{t_f}"
                  r"|\lambda-\widehat{\lambda}(t)|\:dt$")
    ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3])
    ax.set_yticks([1e-11, 1e-10, 1e-8, 1e-7, 1e-5, 1e-4], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_ylim(1e-13, 1e-2)
    ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")

    fig.savefig("figures/finitediff_order.pdf", dpi=300, bbox_inches="tight")


def convergence_interpolator():
    """Figure 4: Convergence in time for single-parameter estimation
    with different number of Fourier modes or pointwise observations.

    Requires the folders data/interpolator_scan and
    data/interpolator_scan_critical_range.
    """
    sr1 = SimulationResults("data/interpolator_scan")
    sr2 = SimulationResults("data/interpolator_scan_critical_range")

    # Plot results.
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,7))

    sr2.results[67].plot(ax=axes[0,0])
    axes[0,0].set_title("Fourier projection, 18 modes")

    sr1.results[9].plot(ax=axes[0,1])
    axes[0,1].set_title("Fourier projection, 21 modes")

    sr2.results[28].plot(ax=axes[1,0])
    axes[1,0].set_title("Cubic interpolation, 40 points")

    sr2.results[55].plot(ax=axes[1,1])
    axes[1,1].set_title("Cubic interpolation, 46 points")

    # Format axes.
    for ax in axes.flat:
        ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
        ax.set_yticks([1e-14,
                       1e-12, 1e-11, 1e-10,
                       1e-8, 1e-7, 1e-6,
                       1e-4, 1e-3, 1e-2,
                       1e0, 1e1],
                      minor=True)
        ax.set_yticklabels([], minor=True)
        ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")
        ax.set_ylim([1e-15, 1e2])
        ax.set_xlim(right=50)
    for ax in axes[-1,:]:
        ax.set_xlabel(r"Time $t$")
    for ax in axes[:,0]:
        ax.set_ylabel("Absolute error")

    for ax in axes.flat:
        for line, i in zip(ax.lines, [1, 6, 9]):
            line.set_color(f"C{i-1:d}")
            line.set_linewidth(1)
            line.set_linestyle(_STYLES[i-1])

    # Legend below the plots.
    fig.subplots_adjust(bottom=.2, wspace=.05)
    labels = [
        r"$|\lambda - \widehat{\lambda}(t)|$",
        r"$||I_h(u(\cdot,t)) - I_h(v(\cdot,t))||$",
        r"$||u(\cdot,t) - v(\cdot,t)||$"
    ]
    leg = axes[0,0].legend(labels, loc="lower center", ncol=3,
                           bbox_to_anchor=(.5,0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(4)

    fig.savefig("figures/convergence_interpolator.pdf",
                dpi=300, bbox_inches="tight")


def _configure_convergence_plot(ax, indices, xmax=50, legend=True):
    """Common settings for convergence Figures."""
    ax.set_xlabel(r"Time $t$")
    ax.set_xlim(right=xmax)
    ax.set_ylabel("Absolute error")
    ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
    ax.set_yticks([1e-14,
                   1e-12, 1e-11, 1e-10,
                   1e-8, 1e-7, 1e-6,
                   1e-4, 1e-3, 1e-2,
                   1e0, 1e1],
                  minor=True)
    ax.set_yticklabels([], minor=True)
    ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")
    ax.set_ylim([1e-15, 1e2])
    ax.set_title(r"Estimation of $" + ", ".join([fr"\lambda_{{{i}}}"
                                                 for i in indices]) + "$")

    labels = []
    if len(indices) == 1:
        indices += [6, 9]
    for line,i in zip(ax.lines, indices):
        fixed = _texlabel(line.get_label())
        line.set_label(fixed)
        line.set_color(f"C{i-1:d}")
        line.set_linestyle(_STYLES[i-1])
        line.set_linewidth(1)
        labels.append(fixed)

    if legend:
        fig = ax.get_figure()
        fig.subplots_adjust(right=.6)
        leg = ax.legend(labels, loc="center left", ncol=1, fontsize="xx-large",
                        bbox_to_anchor=(.625,.5),
                        bbox_transform=fig.transFigure)
        for line in leg.get_lines():
            line.set_linewidth(4)


def _texlabel(lbl):
    """Convert a label to the corresponding TeX equation."""
    if lbl.startswith("true"):
        return r"$||u(\cdot,t) - v(\cdot,t)||$"
    elif lbl.startswith("interp"):
        return r"$||I_h(u(\cdot,t)) - I_h(v(\cdot,t))||$"
    else:
        if not lbl.startswith("lambda"):
            raise ValueError(f"Unrecognized label '{lbl}'")
        num = lbl[-1]
        return fr"$|\lambda_{num} - \widehat{{\lambda}}_{num}(t)|$"


def convergence_multiparam():
    """Figure 5: Convergence in time for multi-parameter estimation.

    Requires the folder data/convergence_multiparam/.
    """
    sr = SimulationResults("data/convergence_multiparam")
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,4.5), sharey=True)

    ax1 = sr.results[0].plot(ax1, params_only=True)
    _configure_convergence_plot(ax1, [1,2,3,4], legend=False)
    ax2 = sr.results[1].plot(ax2, params_only=True)
    _configure_convergence_plot(ax2, [2,4,5], legend=False)
    ax2.set_ylabel("")
    fig.subplots_adjust(bottom=.4, wspace=.05)

    handles = ax1.lines + [ax2.lines[-1]]
    labels = [_texlabel(f"lambda {i}") for i in [1,2,3,4,5]]
    leg = ax1.legend(handles, labels,
                     loc="lower center", ncol=3, fontsize="xx-large",
                     bbox_to_anchor=(.5,0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig("figures/convergence_multiparam.pdf",
                dpi=300, bbox_inches="tight")


def convergence_nonlinearparam():
    """Figures 6: Convergence in time for esimating the nonlinear parameter.

    Requires the folder data/convergence_multiparam/.
    """
    sr = SimulationResults("data/convergence_multiparam")

    ax = sr.results[2].plot(figsize=(12,3), params_only=False)
    _configure_convergence_plot(ax, [5])

    plt.savefig("figures/convergence_nonlinearparam.pdf",
                dpi=300, bbox_inches="tight")


# Main routine ================================================================

def main():
    """Create all plots."""
    init_plt_settings()
    convergence_singleparam()
    mu_alpha_convergence()
    finitedifference_order()
    convergence_interpolator()
    convergence_multiparam()
    convergence_nonlinearparam()


if __name__ == "__main__":
    main()
