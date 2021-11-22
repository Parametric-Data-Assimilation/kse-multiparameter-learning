# plots.py
"""Generate all Figures used in the paper."""
import matplotlib.pyplot as plt

from simulation_results import SimulationResults
import numpy as np


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

    Requires the folder data/explore_alpha/.
    """
    sr = SimulationResults("data/explore_alpha")

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
    axes[0].set_ylabel("Convergence rate")
    for ax in axes:
        ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")

    fig.subplots_adjust(wspace=.05)
    fig.savefig("figures/mu_alpha_convergence.pdf",
                dpi=300, bbox_inches="tight")


def finite_difference_order():
    """Figure 2: Convergence against time step for various FD schemes.

    Requires the folder data/order_diffusion1/.
    """
    sr = SimulationResults("data/order_diffusion1")

    # Extract and plot results.
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    df = sr.get_summary()
    data = df.groupby(["dt","order"])["lambda2_error"].sum().unstack()
    dt = data.index.values
    print("Estimated Orders of Accuracy:")
    for order, mark in zip(data.columns, "osd"):
        estimated_order = np.polyfit(np.log(dt)[order-1:],
                                     np.log(data[order].values[order-1:]),
                                     1)[0]
        print(order, estimated_order)
        ax.loglog(dt, data[order].values,
                  ls='-', lw=1, marker=mark, ms=8, mew=0,
                  label=f"Order {order:d}")

    # Set labels and titles.
    ax.set_xlabel(r"Time step $\delta t$")
    ax.set_ylabel(r"$|\lambda(t_f) - \widehat{\lambda}(t_f)|$")
    ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3])
    ax.set_yticks([1e-11, 1e-10, 1e-8, 1e-7, 1e-5, 1e-4], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_ylim(1e-13, 1e-2)
    ax.set_xlim(ax.get_xlim()[::-1])  # Reverse x axis.
    ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")

    # Legend to the side of the plot.
    fig.subplots_adjust(right=.825)
    ax.legend("123",  # ["First order", "Second order", "Third order"],
              title="FD order", title_fontsize="xx-large",
              loc="center right", ncol=1,
              bbox_to_anchor=(1,.5), bbox_transform=fig.transFigure)

    fig.savefig("figures/finitediff_order.pdf", dpi=300, bbox_inches="tight")


def _configure_convergence_plot(ax, indices, xmax=50, legend=True):
    """Common settings for Figures 3, 4, and 5."""
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
    """Figure 4: Convergence in time for multi-parameter estimation.

    Requires the folder data/sample_multiparam/.
    """
    sr = SimulationResults("data/sample_multiparam")
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
    """Figures 5: Convergence in time for esimating the nonlinear parameter.

    Requires the folder data/sample_multiparam/.
    """
    sr = SimulationResults("data/sample_multiparam")

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
    finite_difference_order()
    convergence_multiparam()
    convergence_nonlinearparam()


if __name__ == "__main__":
    main()
