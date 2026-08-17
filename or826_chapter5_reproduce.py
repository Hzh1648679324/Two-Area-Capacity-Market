from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError as exc:
    raise SystemExit(
        "numba is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc

# -----------------------------------------------------------------------------
# Locked dissertation conventions
# -----------------------------------------------------------------------------
HOURS_PER_WINTER = 3360
N_WINTERS = 7
UNIT_CAPACITY_MW = 500
UNIT_AVAILABILITY = 0.938

C_GRID = list(range(0, 2001, 250))
BACKGROUND_CASES = [
    (500, "higher-risk"),
    (1000, "medium-risk"),
    (2000, "lower-risk"),
]
LOCATION_CASES = ["GB--GB", "GB--Ireland", "Ireland--Ireland"]
DIAGNOSTIC_GRID = [
    0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000,
    2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000,
]
OUTPUT_FILES = [
    "raw_ireland_ireland.csv",
    "controlled_ireland_ireland.csv",
    "controlled_location_cases_medium.csv",
    "diagnostics_medium.csv",
]


@dataclass(frozen=True)
class ModelData:
    gb_net: np.ndarray
    ireland_net: np.ndarray
    ireland_net_float: np.ndarray
    gb_pmf: np.ndarray
    ireland_pmf: np.ndarray
    gb_cdf: np.ndarray


# -----------------------------------------------------------------------------
# Core adequacy and EFC functions used by Chapter 5
# -----------------------------------------------------------------------------
def read_space(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", quotechar='"')


def resolve_data_dir(path: Path) -> Path:
    candidates = [path, path / "data", path / "data" / "data"]
    required = {
        "InterconnectionData_Rescaled.txt",
        "InterconnectionData_peak.txt",
        "GB_anonymised_conv.txt",
        "I_conv.txt",
    }
    for candidate in candidates:
        if candidate.is_dir() and required.issubset({p.name for p in candidate.iterdir()}):
            return candidate
    raise FileNotFoundError(
        "Could not find all four input files. Point --data-dir to the folder "
        "containing InterconnectionData_Rescaled.txt, InterconnectionData_peak.txt, "
        "GB_anonymised_conv.txt and I_conv.txt."
    )


def build_pmf(capacities: Iterable[float], availabilities: Iterable[float]) -> np.ndarray:
    pmf = np.array([1.0], dtype=np.float64)
    for capacity, availability in zip(capacities, availabilities):
        cap = int(round(float(capacity)))
        a = float(availability)
        if cap < 0 or not (0.0 <= a <= 1.0):
            raise ValueError("Invalid conventional-unit capacity or availability.")
        updated = np.zeros(len(pmf) + cap, dtype=np.float64)
        updated[: len(pmf)] += pmf * (1.0 - a)
        updated[cap : cap + len(pmf)] += pmf * a
        pmf = updated
    pmf[pmf < 1e-18] = 0.0
    pmf /= pmf.sum()
    return pmf


def add_unit(
    pmf: np.ndarray,
    capacity_mw: int = UNIT_CAPACITY_MW,
    availability: float = UNIT_AVAILABILITY,
) -> np.ndarray:
    updated = np.zeros(len(pmf) + capacity_mw, dtype=np.float64)
    updated[: len(pmf)] += pmf * (1.0 - availability)
    updated[capacity_mw : capacity_mw + len(pmf)] += pmf * availability
    updated /= updated.sum()
    return updated


def load_data(data_dir: Path) -> ModelData:
    data_dir = resolve_data_dir(data_dir)
    rescaled = read_space(data_dir / "InterconnectionData_Rescaled.txt")
    _ = read_space(data_dir / "InterconnectionData_peak.txt")
    gb_conv = read_space(data_dir / "GB_anonymised_conv.txt")
    ie_conv = read_space(data_dir / "I_conv.txt")

    expected_hours = HOURS_PER_WINTER * N_WINTERS
    if len(rescaled) != expected_hours:
        raise ValueError(f"Expected {expected_hours} hourly observations, found {len(rescaled)}.")
    if rescaled[["Date", "Time"]].duplicated().any():
        raise ValueError("Duplicate Date-Time observations were found.")

    gb_net_float = (rescaled["GBdem_r"] - rescaled["GBwind_r"]).to_numpy(float)
    ie_net_float = (rescaled["Idem_r"] - rescaled["Iwind_r"]).to_numpy(float)

    # Final Chapter 3 convention: map quantities used in the discrete capacity
    # model to the nearest MW before evaluating risk.
    gb_net = np.rint(gb_net_float).astype(np.float64)
    ireland_net = np.rint(ie_net_float).astype(np.float64)

    gb_pmf = build_pmf(gb_conv["Capacity"], gb_conv["Availability"])
    ireland_pmf = build_pmf(ie_conv["Capacity"], ie_conv["Availability"])

    return ModelData(
        gb_net=gb_net,
        ireland_net=ireland_net,
        ireland_net_float=ie_net_float,
        gb_pmf=gb_pmf,
        ireland_pmf=ireland_pmf,
        gb_cdf=np.cumsum(gb_pmf),
    )


@njit(cache=True)
def cdf_less(cdf: np.ndarray, threshold: float) -> float:
    idx = int(np.ceil(threshold)) - 1
    if idx < 0:
        return 0.0
    if idx >= len(cdf):
        return 1.0
    return cdf[idx]


@njit(cache=True)
def cdf_le(cdf: np.ndarray, threshold: float) -> float:
    idx = int(np.floor(threshold))
    if idx < 0:
        return 0.0
    if idx >= len(cdf):
        return 1.0
    return cdf[idx]


@njit(cache=True)
def tail_ge(cdf: np.ndarray, threshold: float) -> float:
    idx_before = int(np.ceil(threshold)) - 1
    if idx_before < 0:
        return 1.0
    if idx_before >= len(cdf):
        return 0.0
    return 1.0 - cdf[idx_before]


@njit(cache=True)
def average_winter_lole_kernel(
    gb_net: np.ndarray,
    ireland_net: np.ndarray,
    gb_cdf: np.ndarray,
    ireland_pmf: np.ndarray,
    ireland_cdf: np.ndarray,
    interconnector_capacity_mw: float,
    gb_firm_capacity_mw: float,
) -> float:
    total = 0.0
    c = interconnector_capacity_mw
    n_ie_states = len(ireland_pmf)

    if c <= 0.0:
        for t in range(len(gb_net)):
            total += cdf_less(gb_cdf, gb_net[t] - gb_firm_capacity_mw)
        return total / N_WINTERS

    for t in range(len(gb_net)):
        ng = gb_net[t] - gb_firm_capacity_mw
        ni = ireland_net[t]
        lolp = 0.0

        p_no_export = cdf_le(ireland_cdf, ni)
        if p_no_export > 0.0:
            lolp += p_no_export * cdf_less(gb_cdf, ng)

        p_full_export = tail_ge(ireland_cdf, ni + c)
        if p_full_export > 0.0:
            lolp += p_full_export * cdf_less(gb_cdf, ng - c)

        start = int(np.floor(ni)) + 1
        end = int(np.ceil(ni + c)) - 1
        if start < 0:
            start = 0
        if end >= n_ie_states:
            end = n_ie_states - 1
        for xi in range(start, end + 1):
            export = xi - ni
            if 0.0 < export < c:
                lolp += ireland_pmf[xi] * cdf_less(gb_cdf, ng - export)
        total += lolp

    # Average the seven winter totals; no 8760/23520 annual scaling.
    return total / N_WINTERS


def average_winter_lole(
    gb_net: np.ndarray,
    ireland_net: np.ndarray,
    gb_cdf: np.ndarray,
    ireland_pmf: np.ndarray,
    c: int | float,
    gb_firm: int | float = 0,
) -> float:
    return float(
        average_winter_lole_kernel(
            np.asarray(gb_net, dtype=np.float64),
            np.asarray(ireland_net, dtype=np.float64),
            np.asarray(gb_cdf, dtype=np.float64),
            np.asarray(ireland_pmf, dtype=np.float64),
            np.asarray(np.cumsum(ireland_pmf), dtype=np.float64),
            float(c),
            float(gb_firm),
        )
    )


def solve_minimum_integer_efc(
    firm_lole: Callable[[int], float],
    target_lole: float,
    max_mw: int = 10000,
) -> tuple[int, float, float]:
    if firm_lole(0) <= target_lole + 1e-12:
        return 0, firm_lole(0), np.nan

    high = 1
    while high < max_mw and firm_lole(high) > target_lole + 1e-12:
        high *= 2
    high = min(high, max_mw)
    if firm_lole(high) > target_lole + 1e-12:
        raise RuntimeError("Could not bracket the EFC within the search interval.")

    low = 0
    while low < high:
        mid = (low + high) // 2
        if firm_lole(mid) <= target_lole + 1e-12:
            high = mid
        else:
            low = mid + 1

    efc = int(low)
    matched = firm_lole(efc)
    previous = firm_lole(efc - 1) if efc > 0 else np.nan
    if efc > 0 and not (
        matched <= target_lole + 1e-10 and previous > target_lole - 1e-10
    ):
        raise RuntimeError(f"Minimum-integer EFC check failed at {efc} MW.")
    return efc, matched, previous


def calibrate_shift(
    data: ModelData,
    c: int,
    target_lole: float,
    max_shift: int = 50000,
) -> tuple[int, float]:
    cache: dict[int, float] = {}

    def risk(shift: int) -> float:
        shift = int(shift)
        if shift not in cache:
            cache[shift] = average_winter_lole(
                data.gb_net + shift,
                data.ireland_net,
                data.gb_cdf,
                data.ireland_pmf,
                c,
            )
        return cache[shift]

    bound = 500
    while bound <= max_shift and not (risk(-bound) <= target_lole <= risk(bound)):
        bound *= 2
    if bound > max_shift:
        raise RuntimeError(f"Could not bracket target LOLE {target_lole} at c={c} MW.")

    low, high = -bound, bound
    while high - low > 1:
        mid = (low + high) // 2
        if risk(mid) < target_lole:
            low = mid
        else:
            high = mid

    candidates = range(low - 2, high + 3)
    best = min(candidates, key=lambda q: (abs(risk(q) - target_lole), abs(q)))
    return int(best), float(risk(best))


# -----------------------------------------------------------------------------
# Chapter 5 portfolio calculations
# -----------------------------------------------------------------------------
def resource_risk(
    data: ModelData,
    gb_net: np.ndarray,
    c: int,
    gb_pmf: np.ndarray | None = None,
    ie_pmf: np.ndarray | None = None,
) -> float:
    gb_pmf = data.gb_pmf if gb_pmf is None else gb_pmf
    ie_pmf = data.ireland_pmf if ie_pmf is None else ie_pmf
    return average_winter_lole(
        gb_net,
        data.ireland_net,
        np.cumsum(gb_pmf),
        ie_pmf,
        c,
    )


def portfolio_metrics(data: ModelData, gb_net: np.ndarray, c: int, loc: str) -> dict[str, float]:
    base = resource_risk(data, gb_net, c)
    gb1 = add_unit(data.gb_pmf)
    gb2 = add_unit(gb1)
    ie1 = add_unit(data.ireland_pmf)
    ie2 = add_unit(ie1)

    if loc == "GB--GB":
        r_a = resource_risk(data, gb_net, c, gb_pmf=gb1)
        r_b = r_a
        r_ab = resource_risk(data, gb_net, c, gb_pmf=gb2)
    elif loc == "GB--Ireland":
        r_a = resource_risk(data, gb_net, c, gb_pmf=gb1)
        r_b = resource_risk(data, gb_net, c, ie_pmf=ie1)
        r_ab = resource_risk(data, gb_net, c, gb_pmf=gb1, ie_pmf=ie1)
    elif loc == "Ireland--Ireland":
        r_a = resource_risk(data, gb_net, c, ie_pmf=ie1)
        r_b = r_a
        r_ab = resource_risk(data, gb_net, c, ie_pmf=ie2)
    else:
        raise ValueError(f"Unknown location case: {loc}")

    cache = {0: base}

    def firm(e: int) -> float:
        if e not in cache:
            cache[e] = average_winter_lole(
                gb_net,
                data.ireland_net,
                data.gb_cdf,
                data.ireland_pmf,
                c,
                e,
            )
        return cache[e]

    e_a = solve_minimum_integer_efc(firm, r_a)[0]
    e_b = solve_minimum_integer_efc(firm, r_b)[0]
    e_ab = solve_minimum_integer_efc(firm, r_ab)[0]

    return {
        "base": base,
        "rA": r_a,
        "rB": r_b,
        "rAB": r_ab,
        "eA": e_a,
        "eB": e_b,
        "eAB": e_ab,
        "delta": e_a + e_b - e_ab,
        "incB": e_ab - e_a,
    }


def compute_outputs(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)

    targets = {c: resource_risk(data, data.gb_net, c) for c, _ in BACKGROUND_CASES}

    print("Calculating raw Ireland--Ireland portfolio results...", flush=True)
    raw_rows = []
    for c in C_GRID:
        raw_rows.append({"c": c, **portfolio_metrics(data, data.gb_net, c, "Ireland--Ireland")})
    raw = pd.DataFrame(raw_rows)

    print("Calculating controlled Ireland--Ireland backgrounds...", flush=True)
    controlled_rows = []
    for reference_c, label in BACKGROUND_CASES:
        target = targets[reference_c]
        for c in C_GRID:
            shift, achieved = calibrate_shift(data, c, target)
            metrics = portfolio_metrics(data, data.gb_net + shift, c, "Ireland--Ireland")
            controlled_rows.append(
                {
                    "label": label,
                    "target": target,
                    "c": c,
                    "shift": shift,
                    "achieved": achieved,
                    **metrics,
                }
            )
    controlled = pd.DataFrame(controlled_rows)

    print("Calculating medium-background location controls...", flush=True)
    location_rows = []
    medium_target = targets[1000]
    for c in C_GRID:
        shift, achieved = calibrate_shift(data, c, medium_target)
        for loc in LOCATION_CASES:
            location_rows.append(
                {
                    "loc": loc,
                    "c": c,
                    "shift": shift,
                    "achieved": achieved,
                    **portfolio_metrics(data, data.gb_net + shift, c, loc),
                }
            )
    locations = pd.DataFrame(location_rows)

    print("Calculating simplified-model state diagnostics...", flush=True)
    a = UNIT_AVAILABILITY
    k = UNIT_CAPACITY_MW
    ie1 = add_unit(data.ireland_pmf, k, a)
    ie2 = add_unit(ie1, k, a)
    ie_det1 = add_unit(data.ireland_pmf, k, 1.0)
    ie_det2 = add_unit(data.ireland_pmf, 2 * k, 1.0)

    maximum_surplus = (len(ie2) - 1) - data.ireland_net_float.min()
    c_infinity = int(np.ceil(maximum_surplus)) + 10
    diagnostic_rows = []

    for c in DIAGNOSTIC_GRID + [c_infinity]:
        shift, _ = calibrate_shift(data, c, medium_target)
        gb_net = data.gb_net + shift
        base = resource_risk(data, gb_net, c)
        r_a = resource_risk(data, gb_net, c, ie_pmf=ie1)
        r_ab = resource_risk(data, gb_net, c, ie_pmf=ie2)
        r_d1 = resource_risk(data, gb_net, c, ie_pmf=ie_det1)
        r_d2 = resource_risk(data, gb_net, c, ie_pmf=ie_det2)

        b_a = base - r_a
        b_ab = base - r_ab
        b_d1 = base - r_d1
        b_d2 = base - r_d2

        overlap = a**2 * b_d1
        complementarity = a**2 * (b_d2 - b_d1)
        gamma = overlap - complementarity

        cache = {0: base}

        def firm(e: int) -> float:
            if e not in cache:
                cache[e] = average_winter_lole(
                    gb_net,
                    data.ireland_net,
                    data.gb_cdf,
                    data.ireland_pmf,
                    c,
                    e,
                )
            return cache[e]

        e_a = solve_minimum_integer_efc(firm, r_a)[0]
        e_ab = solve_minimum_integer_efc(firm, r_ab)[0]
        additive_target = base - 2 * b_a
        e_add = solve_minimum_integer_efc(firm, additive_target)[0]

        delta_curve = 2 * e_a - e_add
        delta_state = e_add - e_ab
        delta = 2 * e_a - e_ab

        diagnostic_rows.append(
            {
                "c": c,
                "shift": shift,
                "base": base,
                "bA": b_a,
                "bAB": b_ab,
                "O": overlap,
                "C": complementarity,
                "gamma": gamma,
                "gamma2": 2 * b_a - b_ab,
                "eA": e_a,
                "eAB": e_ab,
                "eadd": e_add,
                "dcurve": delta_curve,
                "dstate": delta_state,
                "delta": delta,
                "res": delta - delta_curve - delta_state,
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)

    raw.to_csv(output_dir / OUTPUT_FILES[0], index=False)
    controlled.to_csv(output_dir / OUTPUT_FILES[1], index=False)
    locations.to_csv(output_dir / OUTPUT_FILES[2], index=False)
    diagnostics.to_csv(output_dir / OUTPUT_FILES[3], index=False)

    summary = [
        "OR826 Chapter 5 full recomputation",
        "====================================",
        f"Non-binding benchmark c_infinity: {c_infinity} MW",
        f"Maximum possible Irish exportable surplus: {maximum_surplus:.6f} MW",
        "",
        "Medium controlled Ireland--Ireland checks:",
    ]
    medium = controlled[controlled["label"] == "medium-risk"]
    for c in [500, 1000, 1500, 2000]:
        row = medium[medium["c"] == c].iloc[0]
        summary.append(f"c={c}: eA={int(row.eA)}, eAB={int(row.eAB)}, delta={int(row.delta)}")
    nb = diagnostics[diagnostics["c"] == c_infinity].iloc[0]
    summary.extend(
        [
            "",
            "Non-binding diagnostic:",
            f"eA={int(nb.eA)}, eAB={int(nb.eAB)}, delta={int(nb.delta)}",
            f"dstate={int(nb.dstate)}, dcurve={int(nb.dcurve)}",
        ]
    )
    (output_dir / "chapter5_recomputation_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(f"Done. Numerical outputs written to: {output_dir.resolve()}", flush=True)


# -----------------------------------------------------------------------------
# Figure creation
# -----------------------------------------------------------------------------
def create_figures(input_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_dir / OUTPUT_FILES[0])
    controlled = pd.read_csv(input_dir / OUTPUT_FILES[1])
    locations = pd.read_csv(input_dir / OUTPUT_FILES[2])
    diagnostics = pd.read_csv(input_dir / OUTPUT_FILES[3])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(raw["c"], raw["delta"], marker="o", label="Raw background")
    for label in ["higher-risk", "medium-risk", "lower-risk"]:
        group = controlled[controlled["label"] == label]
        target = group["target"].iloc[0]
        ax.plot(group["c"], group["delta"], marker="o", label=f"{target:.4f} h/winter")
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel(r"Non-additivity $\Delta$ (MW)")
    ax.set_xlim(left=0)
    ax.axhline(0, linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(title="GB adequacy background", frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "ch5_ireland_ireland_backgrounds.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label in LOCATION_CASES:
        group = locations[locations["loc"] == label]
        ax.plot(group["c"], group["delta"], marker="o", label=label.replace("--", "–"))
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel(r"Non-additivity $\Delta$ (MW)")
    ax.set_xlim(left=0)
    ax.axhline(0, linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "ch5_location_controls.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    data = diagnostics.copy()
    c_infinity = data["c"].max()
    infinity_x = 5500
    data["x"] = data["c"].where(data["c"] != c_infinity, infinity_x)
    ticks = [0, 1000, 2000, 3000, 4000, 5000, infinity_x]
    labels = ["0", "1000", "2000", "3000", "4000", "5000", r"$c_\infty$"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(data["x"], data["O"], marker="o", label=r"Overlap $\omega(c)$")
    ax.plot(data["x"], data["C"], marker="s", label=r"Complementarity $\kappa(c)$")
    ax.plot(data["x"], data["gamma"], marker="^", label=r"Net physical interaction $\gamma(c)$")
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel("Average-winter LOLE interaction (h/winter)")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(left=0)
    ax.axhline(0, linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "ch5_overlap_complementarity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(data["x"], data["dstate"], marker="o", label=r"State-interaction component $\Delta^{\mathrm{state}}$")
    ax.plot(data["x"], data["dcurve"], marker="s", label=r"Risk-curve component $\Delta^{\mathrm{curve}}$")
    ax.plot(data["x"], data["delta"], marker="^", linewidth=2, label=r"Total $\Delta$")
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel("EFC interaction (MW)")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(left=0)
    ax.axhline(0, linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "ch5_efc_decomposition.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Done. Figures written to: {figure_dir.resolve()}")


# -----------------------------------------------------------------------------
# Verification against locked outputs
# -----------------------------------------------------------------------------
def verify_outputs(generated_dir: Path, reference_dir: Path) -> None:
    passed = True
    for name in OUTPUT_FILES:
        generated_path = generated_dir / name
        reference_path = reference_dir / name
        if not generated_path.exists() or not reference_path.exists():
            print(f"FAIL {name}: missing generated or reference file")
            passed = False
            continue

        generated = pd.read_csv(generated_path)
        reference = pd.read_csv(reference_path)
        if list(generated.columns) != list(reference.columns) or generated.shape != reference.shape:
            print(f"FAIL {name}: schema or row count differs")
            passed = False
            continue

        integer_columns = [
            col for col in generated.columns
            if col in {"c", "shift", "eA", "eB", "eAB", "delta", "incB", "eadd", "dcurve", "dstate", "res"}
        ]
        integer_ok = all(
            np.array_equal(generated[col].to_numpy(), reference[col].to_numpy())
            for col in integer_columns
        )

        numeric_columns = generated.select_dtypes(include=[np.number]).columns
        max_difference = 0.0
        for col in numeric_columns:
            difference = np.nanmax(
                np.abs(generated[col].to_numpy(float) - reference[col].to_numpy(float))
            )
            max_difference = max(max_difference, float(difference))

        text_columns = [col for col in generated.columns if col not in numeric_columns]
        text_ok = all(generated[col].equals(reference[col]) for col in text_columns)

        ok = integer_ok and text_ok and max_difference <= 1e-9
        print(f"{'PASS' if ok else 'FAIL'} {name}: max numeric difference = {max_difference:.3e}")
        passed = passed and ok

    if not passed:
        raise SystemExit(1)
    print("All generated Chapter 5 numerical outputs match the locked reference outputs.")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "All-in-one reproduction script for OR826 Chapter 5: compute numerical "
            "outputs, create figures and verify against locked reference CSV files."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "compute", "plot", "verify"],
        default="all",
        help="Operation to perform (default: all).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("chapter5_outputs"))
    parser.add_argument("--figure-dir", type=Path, default=Path("chapter5_figures"))
    parser.add_argument("--reference-dir", type=Path, default=Path("locked_outputs"))
    parser.add_argument(
        "--plot-input-dir",
        type=Path,
        default=None,
        help="CSV folder used in plot mode; defaults to --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode

    if mode in {"all", "compute"}:
        compute_outputs(args.data_dir.resolve(), args.output_dir.resolve())

    if mode in {"all", "plot"}:
        input_dir = args.plot_input_dir or args.output_dir
        create_figures(input_dir.resolve(), args.figure_dir.resolve())

    if mode in {"all", "verify"}:
        if not args.reference_dir.exists():
            if mode == "all":
                print(
                    f"Reference folder not found at {args.reference_dir.resolve()}; "
                    "calculation and plotting completed, verification skipped."
                )
            else:
                raise FileNotFoundError(
                    f"Reference folder not found: {args.reference_dir.resolve()}"
                )
        else:
            verify_outputs(args.output_dir.resolve(), args.reference_dir.resolve())


if __name__ == "__main__":
    main()
