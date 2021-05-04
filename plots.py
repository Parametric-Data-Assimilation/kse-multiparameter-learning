# plots.py
"""Generate all Figures used in the paper."""
import matplotlib.pyplot as plt

from simulation_results import SimulationResults
import numpy as np

def init_plt_settings(figsize=(12,4), fontsize="xx-large"):
    """Initialize custom matplotlib settings for paper Figures."""
    plt.rc("figure", figsize=figsize, dpi=300)
    plt.rc("axes", titlesize=fontsize, labelsize=fontsize)
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
        ax.set_xlabel(r"Time [s]")
        ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
        ax.set_yticks([1e-12, 1e-11, 1e-10,
                       1e-8, 1e-7, 1e-6,
                       1e-4, 1e-3, 1e-2,
                       1e0, 1e1],
                      minor=True)
        ax.set_yticklabels([], minor=True)
        ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")
        ax.set_ylim([1e-14, 1e2])
    axes[0].set_ylabel(r"Absolute Error")
    axes[0].set_title(r"Strong Relaxation ($\alpha = 1$)")
    # axes[1].set_title(r"Weak Relaxation ($\alpha = 100$)")
    axes[1].set_title(r"No Relaxation")

    # Legend below the plots.
    fig.subplots_adjust(bottom=.3, wspace=.05)
    labels = [r"$|\lambda(t) - \widehat{\lambda}(t)|$",
              r"$||I_h(\mathbf{u}(\cdot,t)) "
                  r"- I_h(\mathbf{v}(\cdot,t))||$",
              r"$||\mathbf{u}(\cdot,t) - \mathbf{v}(\cdot,t)||$"
             ]
    leg = axes[0].legend(labels, loc="lower center", ncol=3,
                         bbox_to_anchor=(.5,0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(3)

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
    ax.set_xlabel(r"Time step $\delta t$")
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


def configure_convergence_plot(ax, xmax=50):
    """Common settings for Figures 3, 4, and 5."""
    ax.set_xlabel(r"Time [s]")
    ax.set_xlim(right=xmax)
    ax.set_ylabel(r"Absolute Error")
    ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
    ax.set_yticks([1e-12, 1e-11, 1e-10,
                   1e-8, 1e-7, 1e-6,
                   1e-4, 1e-3, 1e-2,
                   1e0, 1e1],
                  minor=True)
    ax.set_yticklabels([], minor=True)
    ax.grid(True, which="major", axis="y", ls='--', lw=.25, color="gray")
    ax.set_ylim([1e-15, 1e2])

    labels = []
    for line in ax.lines:
        fixed = fix_label(line.get_label())
        line.set_label(fixed)
        labels.append(fixed)

    fig = ax.get_figure()
    fig.subplots_adjust(bottom=.3, wspace=.05)
    leg = ax.legend(labels, loc="lower center", ncol=len(labels),
                         bbox_to_anchor=(.5,0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(3)


def fix_label(l):
    """Convert a label to valid TeX."""
    if l.startswith("true"):
        return r"$||\mathbf{u}(\cdot,t) - \mathbf{v}(\cdot,t)||$"
    elif l.startswith("interp"):
        return r"$||I_h(\mathbf{u}(\cdot,t)) - I_h(\mathbf{v}(\cdot,t))||$"
    else:
        if not l.startswith("lambda"):
            raise ValueError(f"Unrecognized label '{l}'")
        num = l[-1]
        return fr"$|\lambda_{num} - \widehat{{\lambda}}_{num}(t)|$"


def convergence_multi():
    """Figures 3, 4 and 5: Convergence in time for multi-parameter estimation (and for just the nonlinear parameter)

    Requires the folder data/sample_multiparam
    """
    plt.rc("legend", fontsize="large")

    sr = SimulationResults("data/sample_multiparam")

    # Plot results.
    ax1 = sr.results[0].plot(figsize=(9,3), params_only=True)
    configure_convergence_plot(ax1)
    ax1.set_title(r"Estimation of $\lambda_1, \lambda_2, \lambda_3, \lambda_4$")
    #ax1.legend()
    plt.savefig("figures/convergence_multi_linear.pdf",
                dpi=300, bbox_inches="tight")

    ax2 = sr.results[1].plot(figsize=(9,3), params_only=True)
    configure_convergence_plot(ax2)
    ax2.set_title(r"Estimation of $\lambda_2, \lambda_4, \lambda_5$")
    #ax2.legend()
    plt.savefig("figures/convergence_multi_nonlin.pdf",
                dpi=300, bbox_inches="tight")

    ax3 = sr.results[2].plot(figsize=(9,3), params_only=False)
    configure_convergence_plot(ax3)
    ax3.set_title(r"Estimation of $\lambda_5$")
    #ax3.legend()
    plt.savefig("figures/convergence_single_nonlin.pdf",
                dpi=300, bbox_inches="tight")


# Main routine ================================================================

def main():
    """Create all plots."""
    init_plt_settings()
    convergence_single()
    finite_difference_order()
    convergence_multi()


if __name__ == "__main__":
    main()
