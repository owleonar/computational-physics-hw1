from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

def main():
    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    # read only the first two columns of the supplied power-spectrum table
    data = np.loadtxt(Path("data/lcdm_z0.matter_pk"))
    k_from_file = data[:, 0]
    power_from_file = data[:, 1]

    # interpolate log(P) versus log(k)
    power_spline = CubicSpline(np.log(k_from_file), np.log(power_from_file), bc_type="natural",)

    # Make the evenly spaced k values used in the integral.
    k_max = 25.0
    dk = 0.001
    k = np.arange(0.0, k_max + 0.5 * dk, dk)

    # set P(0)=0, else spline gives the interpolated P(k)
    power = np.zeros_like(k)
    positive_k = k > 0.0
    power[positive_k] = np.exp(power_spline(np.log(k[positive_k])))

    # calculate xi(r) at each r value
    r = np.linspace(40.0, 130.0, 901)
    common_part = k**2 * power
    xi_values = []

    for radius in r:
        # np.sinc is sin(k*r)/(k*r) (gives 1 at k=0)
        sinc_part = np.sinc(k * radius / np.pi)
        integral = simpson(common_part * sinc_part, x=k)
        xi_values.append(integral / (2.0 * np.pi**2))

    xi = np.array(xi_values)

    # peak read from the graph
    bao_peak = 106.0

    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.plot(r, r**2 * xi, color="tab:blue")
    axis.axvline(
        bao_peak,
        color="tab:red",
        linestyle="--",
        label=rf"$r\approx{bao_peak:.0f}\,\mathrm{{Mpc}}/h$",
    )
    axis.set_xlabel(r"$r\;(\mathrm{Mpc}/h)$")
    axis.set_ylabel(r"$r^2\xi(r)\;([\mathrm{Mpc}/h]^2)$")
    axis.set_title("Matter correlation function and BAO peak")
    axis.legend()
    axis.minorticks_on()
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
    fig.savefig(results_directory / "p3-bao-peak.png", dpi=500)
    plt.close(fig)

    print(f"BAO peak by inspection: approximately {bao_peak:.0f} Mpc/h")


if __name__ == "__main__":
    main()
