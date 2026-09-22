from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

F32 = np.float32

# add in float32 from left to right so roundoff remains visible
def sum_left_to_right(values):
    if values.size == 0:
        return F32(0.0)
    return np.cumsum(values, dtype=F32)[-1]

# use h * sum f((i + 1/2)h) over all N bins
def midpoint_rule(number_of_bins):
    n = int(number_of_bins)
    width = F32(1.0) / F32(n)
    midpoints = (np.arange(n, dtype=F32) + F32(0.5)) * width
    heights = np.exp(-midpoints)
    return F32(width * sum_left_to_right(heights))

# use h * [f(0)/2 + f(1)/2 + sum f(ih)]
def trapezoid_rule(number_of_bins):
    n = int(number_of_bins)
    width = F32(1.0) / F32(n)
    inside_points = np.arange(1, n, dtype=F32) * width

    first_height = np.exp(-F32(0.0))
    last_height = np.exp(-F32(1.0))
    inside_heights = np.exp(-inside_points)

    total_height = (F32(0.5) * first_height + F32(0.5) * last_height + sum_left_to_right(inside_heights))
    return F32(width * total_height)

# use (h/3) * [f(0) + f(1) + 4*sum(odd points) + 2*sum(even points)]
def simpson_rule(number_of_bins):
    n = int(number_of_bins)
    
    # safety
    if n % 2 != 0: raise ValueError("Simpson's rule requires an even number of bins.")

    width = F32(1.0) / F32(n)
    odd_points = np.arange(1, n, 2, dtype=F32) * width
    even_points = np.arange(2, n, 2, dtype=F32) * width

    endpoint_heights = np.exp(-F32(0.0)) + np.exp(-F32(1.0))
    odd_heights = np.exp(-odd_points)
    even_heights = np.exp(-even_points)

    weighted_height = (endpoint_heights + F32(4.0) * sum_left_to_right(odd_heights) + F32(2.0) * sum_left_to_right(even_heights))
    return F32(width * weighted_height / F32(3.0))


def main():
    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    # powers of two keep N even and cover both sides of the error minimum
    powers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20, 21])
    bin_counts = 2**powers
    exact_integral = 1.0 - np.exp(-1.0)

    midpoint_answers = []
    trapezoid_answers = []
    simpson_answers = []

    # calculate all three rules for every chosen value of N
    for n in bin_counts:
        midpoint_answers.append(float(midpoint_rule(n)))
        trapezoid_answers.append(float(trapezoid_rule(n)))
        simpson_answers.append(float(simpson_rule(n)))

    midpoint_answers = np.array(midpoint_answers)
    trapezoid_answers = np.array(trapezoid_answers)
    simpson_answers = np.array(simpson_answers)

    midpoint_errors = np.abs(midpoint_answers - exact_integral) / exact_integral
    trapezoid_errors = np.abs(trapezoid_answers - exact_integral) / exact_integral
    simpson_errors = np.abs(simpson_answers - exact_integral) / exact_integral

    # first omitted terms give the expected truncation-error scaling guides
    n64 = bin_counts.astype(float)
    midpoint_guide = 1.0 / (24.0 * n64**2)
    trapezoid_guide = 1.0 / (12.0 * n64**2)
    simpson_guide = 1.0 / (180.0 * n64**4)

    # stop each dotted line before it falls below the float32 error floor
    show_midpoint_guide = midpoint_guide >= 3.0e-8
    show_trapezoid_guide = trapezoid_guide >= 3.0e-8
    show_simpson_guide = simpson_guide >= 3.0e-8

    fig, axis = plt.subplots(figsize=(7.5, 5.5))

    axis.loglog(
        bin_counts,
        midpoint_errors,
        "o-",
        color="tab:blue",
        markersize=3,
        label="midpoint",
    )
    axis.loglog(
        bin_counts[show_midpoint_guide],
        midpoint_guide[show_midpoint_guide],
        ":",
        color="tab:blue",
        linewidth=2,
    )

    axis.loglog(
        bin_counts,
        trapezoid_errors,
        "o-",
        color="tab:orange",
        markersize=3,
        label="trapezoid",
    )
    axis.loglog(
        bin_counts[show_trapezoid_guide],
        trapezoid_guide[show_trapezoid_guide],
        ":",
        color="tab:orange",
        linewidth=2,
    )

    axis.loglog(
        bin_counts,
        simpson_errors,
        "o-",
        color="tab:green",
        markersize=3,
        label="Simpson",
    )
    axis.loglog(
        bin_counts[show_simpson_guide],
        simpson_guide[show_simpson_guide],
        ":",
        color="tab:green",
        linewidth=2,
    )

    axis.set_xlabel(r"$N$ (number of bins)")
    axis.set_ylabel(r"$\epsilon$ (relative error)")
    axis.set_title("Relative Error in Numerical Integration for " + r"$I=\int_0^1 e^{-t}\,dt$")
    axis.set_xlim(bin_counts[0], bin_counts[-1])
    axis.text(
        0.1,
        0.96,
        "roundoff " + r"$\propto \ N$ (all methods, worst case)" + "\n"
        + "truncation (dotted) "
        + r"$\propto \ N^{-2}$ (midpoint, trapezoid), $N^{-4}$ (Simpson)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axis.legend(fontsize=8, loc="lower left")
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=11,
        length=10,
        width=1.2,)

    axis.tick_params(
        axis="both",
        which="minor",
        length=5,
        width=1.0,)
    fig.tight_layout()
    fig.savefig(results_directory / "p2-integration-relative-error.png", dpi=500)
    plt.close(fig)

    print(f"Exact integral: {exact_integral:.12e}")
    print(f"Largest N: {bin_counts[-1]:,}")


if __name__ == "__main__":
    main()
