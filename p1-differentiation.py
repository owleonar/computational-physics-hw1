from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


F32 = np.float32

# use [f(x + h) - f(x)] / h
def forward_difference(function, x, h):
    x = F32(x)
    h = F32(h)
    return F32((function(F32(x + h)) - function(x)) / h)

# use [f(x + h) - f(x - h)] / (2h)
def centered_difference(function, x, h):
    x = F32(x)
    h = F32(h)
    numerator = function(F32(x + h)) - function(F32(x - h))
    return F32(numerator / (F32(2.0) * h))

# combine centered differences at h and 2h
def extrapolated_difference(function, x, h):
    h = F32(h)
    derivative_h = centered_difference(function, x, h)
    derivative_twice_h = centered_difference(function, x, F32(F32(2.0) * h))
    numerator = F32(4.0) * derivative_h - derivative_twice_h
    return F32(numerator / F32(3.0))

# calculate and plot all three errors for one function and one x
def plot_one_panel(axis, function_name, function, x, h_values):
    # exact derivatives are needed for the errors and truncation-error scaling estimates
    if function_name == "cos":
        function_value = np.cos(x)
        exact_derivative = -np.sin(x)
        second_derivative = -np.cos(x)
        third_derivative = np.sin(x)
        fifth_derivative = -np.sin(x)
    else:
        function_value = np.exp(x)
        exact_derivative = np.exp(x)
        second_derivative = np.exp(x)
        third_derivative = np.exp(x)
        fifth_derivative = np.exp(x)

    forward_errors = []
    centered_errors = []
    extrapolated_errors = []

    # try every step size with each formula
    for h in h_values:
        forward_answer = forward_difference(function, x, h)
        centered_answer = centered_difference(function, x, h)
        extrapolated_answer = extrapolated_difference(function, x, h)

        forward_errors.append(abs(float(forward_answer) - exact_derivative) / abs(exact_derivative))
        centered_errors.append(abs(float(centered_answer) - exact_derivative) / abs(exact_derivative))
        extrapolated_errors.append(abs(float(extrapolated_answer) - exact_derivative) / abs(exact_derivative))

    forward_errors = np.array(forward_errors)
    centered_errors = np.array(centered_errors)
    extrapolated_errors = np.array(extrapolated_errors)

    # first omitted Taylor terms of the difference formula give the truncation-error scaling guide estimates
    h64 = h_values.astype(float)
    forward_guide = abs(second_derivative / (2.0 * exact_derivative)) * h64
    centered_guide = abs(third_derivative / (6.0 * exact_derivative)) * h64**2
    extrapolated_guide = abs(fifth_derivative / (30.0 * exact_derivative)) * h64**4

    # only draw a guide where truncation error is larger than roundoff
    epsilon = np.finfo(F32).eps
    basic_roundoff = epsilon * abs(function_value) / (h64 * abs(exact_derivative))
    show_forward_guide = forward_guide >= 2.0 * basic_roundoff
    show_centered_guide = centered_guide >= basic_roundoff
    show_extrapolated_guide = extrapolated_guide >= 1.5 * basic_roundoff

    axis.loglog(h_values, forward_errors, color="tab:blue", label="forward")
    axis.loglog(
        h_values[show_forward_guide],
        forward_guide[show_forward_guide],
        ":",
        color="tab:blue",
        linewidth=1.5,
    )

    axis.loglog(h_values, centered_errors, color="tab:orange", label="centered")
    axis.loglog(
        h_values[show_centered_guide],
        centered_guide[show_centered_guide],
        ":",
        color="tab:orange",
        linewidth=1.5,
    )

    axis.loglog(
        h_values,
        extrapolated_errors,
        color="tab:green",
        label="extrapolated",
    )
    axis.loglog(
        h_values[show_extrapolated_guide],
        extrapolated_guide[show_extrapolated_guide],
        ":",
        color="tab:green",
        linewidth=1.5,
    )

    best_forward = np.argmin(forward_errors)
    best_centered = np.argmin(centered_errors)
    best_extrapolated = np.argmin(extrapolated_errors)

    print(
        f"{function_name}(x), x={x:g}, forward: "
        f"best h={h_values[best_forward]:.3e}, "
        f"error={forward_errors[best_forward]:.3e}"
    )
    print(
        f"{function_name}(x), x={x:g}, centered: "
        f"best h={h_values[best_centered]:.3e}, "
        f"error={centered_errors[best_centered]:.3e}"
    )
    print(
        f"{function_name}(x), x={x:g}, extrapolated: "
        f"best h={h_values[best_extrapolated]:.3e}, "
        f"error={extrapolated_errors[best_extrapolated]:.3e}"
    )

    function_label = r"\cos(x)" if function_name == "cos" else r"e^x"
    axis.set_title(rf"$d{function_label}/dx$ at $x={x:g}$")
    axis.set_xlabel(r"$h$ (step size)")
    axis.set_ylabel(r"$\epsilon$ (relative error)")
    axis.set_xlim(h_values[0], h_values[-1])
    axis.text(
        0.03,
        0.96,
        "roundoff " + r"$\propto \ h^{-1}$ (all methods)" + "\n"
        + "truncation (dotted) " + r"$\propto \ h \ \mathrm{(forward)},\ h^2 \ \mathrm{(centered)},\ h^4 \ \mathrm{(extrapolated)}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
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


def main():
    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    # log spacing shows small and large step sizes inexpensively
    h_values = np.unique(np.logspace(-6, 0, 61).astype(F32))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    plot_one_panel(axes[0, 0], "cos", np.cos, 0.1, h_values)
    plot_one_panel(axes[0, 1], "cos", np.cos, 10.0, h_values)
    plot_one_panel(axes[1, 0], "exp", np.exp, 0.1, h_values)
    plot_one_panel(axes[1, 1], "exp", np.exp, 10.0, h_values)

    fig.suptitle("Relative Error in Finite-Difference Numerical Differentiation")
    fig.tight_layout()
    fig.savefig(results_directory / "p1-differentiation-relative-error.png", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    main()
