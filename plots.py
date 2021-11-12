# plots.py
"""Generate all Figures used in the paper."""
import matplotlib.pyplot as plt

from simulation_results import SimulationResults
import numpy as np


_STYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-.")


def init_plt_settings(figsize=(12,4), fontsize="xx-large"):
    """Initialize custom matplotlib settings for paper Figures."""
    plt.rc("figure", figsize=figsize, dpi=300)
    plt.rc("axes", titlesize=fontsize, labelsize=fontsize, linewidth=.5)
    plt.rc("legend", fontsize=fontsize, edgecolor="none", frameon=False)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")


# Routines for each figure ====================================================

def convergence_single():
    """Figure 1: Convergence in time for single-parameter estimation.

    Requires the folder data/explore_alpha/.
    """
    sr = SimulationResults("data/explore_alpha")

    # Plot results.
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
    sr.results[0].plot(ax=axes[0])
    # sr.results[1].plot(ax=axes[1])
    sr.results[2].plot(ax=axes[1])

    # Set labels and titles.
    for ax in axes:
        ax.set_xlabel(r"Time $t$")
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
    axes[0].set_ylabel(r"Absolute Error")
    axes[0].set_title(r"With Relaxation ($\alpha = 1$)")
    # axes[1].set_title(r"Weak Relaxation ($\alpha = 100$)")
    axes[1].set_title(r"Without Relaxation")

    for ax in axes:
        for line, i in zip(ax.lines, [1, 6, 9]):
            line.set_color(f"C{i-1:d}")
            line.set_linewidth(1)
            line.set_linestyle(_STYLES[i-1])

    # Legend below the plots.
    fig.subplots_adjust(bottom=.3, wspace=.05)
    labels = [r"$|\lambda - \widehat{\lambda}(t)|$",
              r"$||I_h(u(\cdot,t)) - I_h(v(\cdot,t))||$",
              r"$||u(\cdot,t) - v(\cdot,t)||$"
             ]
    leg = axes[0].legend(labels, loc="lower center", ncol=3,
                         bbox_to_anchor=(.5,0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(4)

    plt.savefig("figures/convergence_single.pdf", dpi=300, bbox_inches="tight")


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
        estimated_order = np.polyfit(np.log(dt)[order-1:], np.log(data[order].values[order-1:]), 1)[0]
        print(order, estimated_order)
        ax.loglog(dt, data[order].values,
                  ls='-', lw=1, marker=mark, ms=8, mew=0,
                  label=f"Order {order:d}")

    # Set labels and titles.
    ax.set_xlabel(r"Time Step $\delta t$")
    ax.set_ylabel(r"$|\lambda(t_f) - \widehat{\lambda}(t_f)|$")
    ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3])
    ax.set_yticks([1e-11, 1e-10, 1e-8, 1e-7, 1e-5, 1e-4], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_ylim(1e-13, 1e-2)
    ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")

    # Legend to the side of the plot.
    fig.subplots_adjust(right=.825)
    labels = ["First order", "Second order", "Third order"]
    ax.legend("123",
              title="FD order", title_fontsize="xx-large",
              loc="center right", ncol=1,
              bbox_to_anchor=(1,.5), bbox_transform=fig.transFigure)

    plt.savefig("figures/finitediff_order.pdf", dpi=300, bbox_inches="tight")


def configure_convergence_plot(ax, indices, xmax=50, legend=True):
    """Common settings for Figures 3, 4, and 5."""
    ax.set_xlabel(r"Time $t$")
    ax.set_xlim(right=xmax)
    ax.set_ylabel(r"Absolute Error")
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
    ax.set_title(r"Estimation of $" + ", ".join([f"\lambda_{{{i}}}"
                                                 for i in indices]) + "$")

    labels = []
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-."]
    if len(indices) == 1:
        indices += [6, 9]
    for line,i in zip(ax.lines, indices):
        fixed = get_texlabel(line.get_label())
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


def get_texlabel(l):
    """Convert a label to the corresponding TeX equation."""
    if l.startswith("true"):
        return r"$||u(\cdot,t) - v(\cdot,t)||$"
    elif l.startswith("interp"):
        return r"$||I_h(u(\cdot,t)) - I_h(v(\cdot,t))||$"
    else:
        if not l.startswith("lambda"):
            raise ValueError(f"Unrecognized label '{l}'")
        num = l[-1]
        return fr"$|\lambda_{num} - \widehat{{\lambda}}_{num}(t)|$"


def convergence_multi():
    """Figures 3, 4 and 5: Convergence in time for multi-parameter estimation (and for just the nonlinear parameter)

    Requires the folder data/sample_multiparam/.
    """
    sr = SimulationResults("data/sample_multiparam")

    # Plot results.
    ax1 = sr.results[0].plot(figsize=(9,3), params_only=True)
    configure_convergence_plot(ax1, [1,2,3,4])
    plt.savefig("figures/convergence_multi_linear.pdf",
                dpi=300, bbox_inches="tight")

    ax2 = sr.results[1].plot(figsize=(9,3), params_only=True)
    configure_convergence_plot(ax2, [2,4,5])
    plt.savefig("figures/convergence_multi_nonlin.pdf",
                dpi=300, bbox_inches="tight")

    ax3 = sr.results[2].plot(figsize=(9,3), params_only=False)
    configure_convergence_plot(ax3, [5])
    plt.savefig("figures/convergence_single_nonlin.pdf",
                dpi=300, bbox_inches="tight")


def convergence_aligned():
    """New Figure 3 (combining old Figures 3 and 4)."""
    sr = SimulationResults("data/sample_multiparam")
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,5.25), sharey=True)

    ax1 = sr.results[0].plot(ax1, params_only=True)
    configure_convergence_plot(ax1, [1,2,3,4], legend=False)
    # plt.savefig("figures/convergence_multi_linear.pdf",
    #             dpi=300, bbox_inches="tight")
    ax2 = sr.results[1].plot(ax2, params_only=True)
    configure_convergence_plot(ax2, [2,4,5], legend=False)
    ax2.set_ylabel("")
    fig.subplots_adjust(bottom=.33, wspace=.05)

    handles = ax1.lines + [ax2.lines[-1]]
    labels = [get_texlabel(f"lambda {i}") for i in [1,2,3,4,5]]
    leg = ax1.legend(handles, labels,
                     loc="lower center", ncol=3, fontsize="xx-large",
                     bbox_to_anchor=(.5,0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(4)
    plt.savefig("figures/convergence_multi_both.pdf",
                dpi=300, bbox_inches="tight")

# Main routine ================================================================

def main():
    """Create all plots."""
    init_plt_settings()
    convergence_single()
    finite_difference_order()
    convergence_multi()
    convergence_aligned()


if __name__ == "__main__":
    main()
