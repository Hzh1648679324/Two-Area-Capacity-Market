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
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numba is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc

HOURS_PER_WINTER = 3360
N_WINTERS = 7
UNIT_CAPACITY_MW = 500
UNIT_AVAILABILITY = 0.938
IRISH_WIND_RATING_MW = 3000.0

SELECTED_C_GRID = [0, 500, 1000, 1500, 2000]
CONTROLLED_C_GRID = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
TARGET_REFERENCE_C_MW = [500, 1000, 2000]
ADDITIONAL_WIND_C_GRID = [0, 500, 1000, 1500, 2000, 5000]
ADDITIONAL_WIND_U_MW = [100, 500, 1000, 2000]
WHOLE_WIND_EFC_GRID = [
    0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500,
    2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 7500,
]
QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]


@dataclass(frozen=True)
class ModelData:
    gb_net: np.ndarray
    ireland_net: np.ndarray
    ireland_net_float: np.ndarray
    ireland_demand: np.ndarray
    ireland_wind: np.ndarray
    gb_pmf: np.ndarray
    ireland_pmf: np.ndarray
    gb_cdf: np.ndarray
    ireland_cdf: np.ndarray


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
    _ = read_space(data_dir / "InterconnectionData_peak.txt")  # retained for input validation
    gb_conv = read_space(data_dir / "GB_anonymised_conv.txt")
    ie_conv = read_space(data_dir / "I_conv.txt")

    expected_hours = HOURS_PER_WINTER * N_WINTERS
    if len(rescaled) != expected_hours:
        raise ValueError(f"Expected {expected_hours} hourly observations, found {len(rescaled)}.")
    if rescaled[["Date", "Time"]].duplicated().any():
        raise ValueError("Duplicate Date-Time observations were found.")

    gb_net_float = (rescaled["GBdem_r"] - rescaled["GBwind_r"]).to_numpy(float)
    ie_net_float = (rescaled["Idem_r"] - rescaled["Iwind_r"]).to_numpy(float)
    ie_demand_float = rescaled["Idem_r"].to_numpy(float)
    ie_wind = rescaled["Iwind_r"].to_numpy(float)

    # Locked Chapter 3 convention: quantities used in the discrete capacity model
    # are mapped to the nearest MW before risk is evaluated.
    gb_net = np.rint(gb_net_float).astype(np.float64)
    ireland_net = np.rint(ie_net_float).astype(np.float64)
    ireland_demand = np.rint(ie_demand_float).astype(np.float64)

    gb_pmf = build_pmf(gb_conv["Capacity"], gb_conv["Availability"])
    ireland_pmf = build_pmf(ie_conv["Capacity"], ie_conv["Availability"])

    return ModelData(
        gb_net=gb_net,
        ireland_net=ireland_net,
        ireland_net_float=ie_net_float,
        ireland_demand=ireland_demand,
        ireland_wind=ie_wind,
        gb_pmf=gb_pmf,
        ireland_pmf=ireland_pmf,
        gb_cdf=np.cumsum(gb_pmf),
        ireland_cdf=np.cumsum(ireland_pmf),
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

    # Week 1-5 definition: sum hourly LOLP over each winter and average the
    # seven winter totals. No 8760/23520 scaling is applied.
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


def conventional_results(data: ModelData) -> tuple[pd.DataFrame, pd.DataFrame]:
    gb_plus = add_unit(data.gb_pmf)
    ie_plus = add_unit(data.ireland_pmf)

    raw_rows: list[dict[str, float]] = []
    raw_risk_by_c: dict[int, float] = {}
    for c in SELECTED_C_GRID:
        base = average_winter_lole(
            data.gb_net, data.ireland_net, data.gb_cdf, data.ireland_pmf, c
        )
        raw_risk_by_c[c] = base
        gb_unit_risk = average_winter_lole(
            data.gb_net, data.ireland_net, np.cumsum(gb_plus), data.ireland_pmf, c
        )
        ie_unit_risk = average_winter_lole(
            data.gb_net, data.ireland_net, data.gb_cdf, ie_plus, c
        )
        firm_cache: dict[int, float] = {0: base}

        def firm(e: int) -> float:
            if e not in firm_cache:
                firm_cache[e] = average_winter_lole(
                    data.gb_net,
                    data.ireland_net,
                    data.gb_cdf,
                    data.ireland_pmf,
                    c,
                    e,
                )
            return firm_cache[e]

        efc_gb, matched_gb, _ = solve_minimum_integer_efc(firm, gb_unit_risk)
        efc_ie, matched_ie, _ = solve_minimum_integer_efc(firm, ie_unit_risk)
        raw_rows.append(
            {
                "interconnector_capacity_MW": c,
                "baseline_GB_LOLE_h_per_winter": base,
                "EFC_500MW_GB_unit_MW": efc_gb,
                "EFC_500MW_Irish_unit_MW": efc_ie,
                "GB_unit_capacity_credit_percent": 100.0 * efc_gb / UNIT_CAPACITY_MW,
                "Irish_unit_capacity_credit_percent": 100.0 * efc_ie / UNIT_CAPACITY_MW,
                "matched_firm_LOLE_GB_unit_h_per_winter": matched_gb,
                "matched_firm_LOLE_Irish_unit_h_per_winter": matched_ie,
            }
        )

    # Chapter 4 uses the raw baseline risks at c=500, 1000 and 2000 MW as
    # controlled-background targets, matching the Week 5 design.
    targets = {c: raw_risk_by_c[c] for c in TARGET_REFERENCE_C_MW}
    labels = {500: "higher-risk", 1000: "medium-risk", 2000: "lower-risk"}

    controlled_rows: list[dict[str, float | str]] = []
    for reference_c in TARGET_REFERENCE_C_MW:
        target = targets[reference_c]
        label = labels[reference_c]
        for c in CONTROLLED_C_GRID:
            shift, achieved = calibrate_shift(data, c, target)
            shifted_gb_net = data.gb_net + shift
            resource_risk = average_winter_lole(
                shifted_gb_net, data.ireland_net, data.gb_cdf, ie_plus, c
            )
            firm_cache: dict[int, float] = {0: achieved}

            def firm(e: int) -> float:
                if e not in firm_cache:
                    firm_cache[e] = average_winter_lole(
                        shifted_gb_net,
                        data.ireland_net,
                        data.gb_cdf,
                        data.ireland_pmf,
                        c,
                        e,
                    )
                return firm_cache[e]

            efc, matched, _ = solve_minimum_integer_efc(firm, resource_risk)
            controlled_rows.append(
                {
                    "interconnector_capacity_MW": c,
                    "background_label": label,
                    "target_baseline_GB_LOLE_h_per_winter": target,
                    "achieved_baseline_GB_LOLE_h_per_winter": achieved,
                    "target_error_h_per_winter": achieved - target,
                    "GB_net_demand_shift_MW": shift,
                    "EFC_500MW_Irish_unit_MW": efc,
                    "Irish_unit_capacity_credit_percent": 100.0 * efc / UNIT_CAPACITY_MW,
                    "matched_firm_LOLE_h_per_winter": matched,
                }
            )

    return pd.DataFrame(raw_rows), pd.DataFrame(controlled_rows)


def additional_wind_results(data: ModelData) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for c in ADDITIONAL_WIND_C_GRID:
        base = average_winter_lole(
            data.gb_net, data.ireland_net, data.gb_cdf, data.ireland_pmf, c
        )
        firm_cache: dict[int, float] = {0: base}

        def firm(e: int) -> float:
            if e not in firm_cache:
                firm_cache[e] = average_winter_lole(
                    data.gb_net,
                    data.ireland_net,
                    data.gb_cdf,
                    data.ireland_pmf,
                    c,
                    e,
                )
            return firm_cache[e]

        # Reconstruct from the unrounded profile, then map the modified net
        # demand to the nearest MW, as specified in locked Chapter 3.
        base_ie_net_float = data.ireland_net_float
        for u in ADDITIONAL_WIND_U_MW:
            modified_ie_net = np.rint(
                base_ie_net_float - data.ireland_wind * (u / IRISH_WIND_RATING_MW)
            ).astype(np.float64)
            resource_risk = average_winter_lole(
                data.gb_net, modified_ie_net, data.gb_cdf, data.ireland_pmf, c
            )
            efc, matched, _ = solve_minimum_integer_efc(firm, resource_risk)
            rows.append(
                {
                    "interconnector_capacity_MW": c,
                    "additional_wind_capacity_MW": u,
                    "baseline_GB_LOLE_h_per_winter": base,
                    "GB_LOLE_with_additional_wind_h_per_winter": resource_risk,
                    "EFC_additional_Irish_wind_MW": efc,
                    "matched_firm_LOLE_h_per_winter": matched,
                    "capacity_credit_percent": 100.0 * efc / u,
                }
            )
    return pd.DataFrame(rows)


def effective_unlimited_c(data: ModelData) -> int:
    # The benchmark must be non-binding both without and with Irish wind.
    # The larger possible surplus occurs in the with-wind case, so use the
    # minimum Irish net demand rather than the minimum gross demand.
    maximum_surplus = float(
        np.max((len(data.ireland_pmf) - 1) - data.ireland_net_float)
    )
    return int(np.ceil(maximum_surplus)) + 10


def whole_wind_results(data: ModelData) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_inf = effective_unlimited_c(data)
    c_grid = sorted(set(WHOLE_WIND_EFC_GRID + [c_inf]))
    pair_cache: dict[int, tuple[float, float]] = {}

    def risks(c: int) -> tuple[float, float]:
        if c not in pair_cache:
            without_wind = average_winter_lole(
                data.gb_net,
                data.ireland_demand,
                data.gb_cdf,
                data.ireland_pmf,
                c,
            )
            with_wind = average_winter_lole(
                data.gb_net,
                data.ireland_net,
                data.gb_cdf,
                data.ireland_pmf,
                c,
            )
            pair_cache[c] = (without_wind, with_wind)
        return pair_cache[c]

    no_wind_inf, with_wind_inf = risks(c_inf)
    benefit_inf = no_wind_inf - with_wind_inf

    rows: list[dict[str, float | str | bool]] = []
    for c in c_grid:
        without_wind, with_wind = risks(c)
        firm_cache: dict[int, float] = {0: without_wind}

        def firm(e: int) -> float:
            if e not in firm_cache:
                firm_cache[e] = average_winter_lole(
                    data.gb_net,
                    data.ireland_demand,
                    data.gb_cdf,
                    data.ireland_pmf,
                    c,
                    e,
                )
            return firm_cache[e]

        efc, matched, _ = solve_minimum_integer_efc(firm, with_wind)
        benefit = without_wind - with_wind
        fraction = benefit / benefit_inf
        rows.append(
            {
                "interconnector_capacity_MW": c,
                "capacity_label": "unlimited" if c == c_inf else str(c),
                "is_unlimited_self_priority_case": c == c_inf,
                "GB_LOLE_without_Irish_wind_h_per_winter": without_wind,
                "GB_LOLE_with_Irish_wind_h_per_winter": with_wind,
                "whole_fleet_wind_LOLE_benefit_h_per_winter": benefit,
                "EFC_whole_Irish_wind_fleet_MW": efc,
                "capacity_credit_percent_of_3000MW": 100.0 * efc / IRISH_WIND_RATING_MW,
                "matched_firm_LOLE_h_per_winter": matched,
                "accessible_fraction_of_unlimited_benefit": fraction,
                "blocked_fraction_of_unlimited_benefit": max(0.0, 1.0 - fraction),
            }
        )

    df = pd.DataFrame(rows)
    efc_inf = float(
        df.loc[df["is_unlimited_self_priority_case"], "EFC_whole_Irish_wind_fleet_MW"].iloc[0]
    )
    df["EFC_fraction_of_unlimited"] = df["EFC_whole_Irish_wind_fleet_MW"] / efc_inf

    qrows: list[dict[str, float]] = []
    for q in QUANTILES:
        low, high = 0, c_inf
        while low < high:
            mid = (low + high) // 2
            no_wind, with_wind = risks(mid)
            fraction = (no_wind - with_wind) / benefit_inf
            if fraction + 1e-13 >= q:
                high = mid
            else:
                low = mid + 1
        no_wind, with_wind = risks(low)
        qrows.append(
            {
                "weighted_quantile": q,
                "GB_deficit_threshold_MW": low,
                "achieved_accessible_fraction": (no_wind - with_wind) / benefit_inf,
            }
        )

    return df, pd.DataFrame(qrows)


def create_figures(
    output_dir: Path,
    raw: pd.DataFrame,
    controlled: pd.DataFrame,
    additional_wind: pd.DataFrame,
    whole_wind: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        raw["interconnector_capacity_MW"],
        raw["EFC_500MW_GB_unit_MW"],
        marker="o",
        label="Resource in GB",
    )
    ax.plot(
        raw["interconnector_capacity_MW"],
        raw["EFC_500MW_Irish_unit_MW"],
        marker="s",
        label="Resource in Ireland",
    )
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel("EFC in GB (MW)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ch4_conventional_location_efc.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, group in controlled.groupby("background_label", sort=False):
        target = group["target_baseline_GB_LOLE_h_per_winter"].iloc[0]
        ax.plot(
            group["interconnector_capacity_MW"],
            group["EFC_500MW_Irish_unit_MW"],
            marker="o",
            label=f"{target:.4f} h/winter",
        )
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel("EFC of 500 MW Irish unit (MW)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Controlled GB LOLE", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ch4_controlled_background_efc.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for c, group in additional_wind.groupby("interconnector_capacity_MW"):
        if c in [500, 1000, 1500, 2000, 5000]:
            ax.plot(
                group["additional_wind_capacity_MW"],
                group["EFC_additional_Irish_wind_MW"],
                marker="o",
                label=f"c = {c} MW",
            )
    ax.set_xlabel("Additional Irish wind capacity (MW)")
    ax.set_ylabel("EFC in GB (MW)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "ch4_additional_wind_efc.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    finite = whole_wind[~whole_wind["is_unlimited_self_priority_case"]].copy()
    unlimited = whole_wind[whole_wind["is_unlimited_self_priority_case"]].iloc[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        finite["interconnector_capacity_MW"],
        finite["EFC_whole_Irish_wind_fleet_MW"],
        marker="o",
    )
    ax.axhline(
        unlimited["EFC_whole_Irish_wind_fleet_MW"],
        linestyle="--",
        label=(
            "Unlimited self-priority: "
            f"{int(unlimited['EFC_whole_Irish_wind_fleet_MW'])} MW"
        ),
    )
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel("Whole-fleet EFC in GB (MW)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ch4_whole_wind_efc.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        finite["interconnector_capacity_MW"],
        100.0 * finite["accessible_fraction_of_unlimited_benefit"],
        marker="o",
    )
    ax.axhline(95, linestyle="--", linewidth=1, label="95% of unlimited benefit")
    ax.axhline(99, linestyle=":", linewidth=1, label="99% of unlimited benefit")
    ax.set_xlabel("Interconnector capacity (MW)")
    ax.set_ylabel("Accessible share of unlimited wind benefit (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ch4_wind_accessible_fraction.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    raw: pd.DataFrame,
    controlled: pd.DataFrame,
    additional_wind: pd.DataFrame,
    whole_wind: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    selected_whole = whole_wind[
        whole_wind["interconnector_capacity_MW"].isin([0, 500, 1000, 1500, 2000, 5000])
        | whole_wind["is_unlimited_self_priority_case"]
    ]
    text = [
        "OR826 Chapter 4 reproduction summary",
        "======================================",
        "Risk metric: average GB LOLE across seven 20-week winters.",
        "No 8760/23520 annual scaling is applied.",
        "EFC: minimum integer MW of perfectly reliable GB firm capacity.",
        "",
        "Conventional location results:",
        raw.to_string(index=False),
        "",
        "Controlled-background maximum absolute calibration error:",
        f"{controlled['target_error_h_per_winter'].abs().max():.12f} h/winter",
        "",
        "Additional wind results:",
        additional_wind.to_string(index=False),
        "",
        "Selected whole-fleet wind results:",
        selected_whole.to_string(index=False),
        "",
        "Wind-helpful thresholds:",
        thresholds.to_string(index=False),
    ]
    (output_dir / "chapter4_reproduction_summary.txt").write_text(
        "\n".join(text), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the numerical outputs and figures used in OR826 Chapter 4."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing the four OR826 data files (default: ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chapter4_outputs"),
        help="Output folder for CSVs and figures (default: ./chapter4_outputs).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir.resolve())

    print("Calculating conventional-resource results...", flush=True)
    raw, controlled = conventional_results(data)
    print("Calculating additional-wind results...", flush=True)
    additional_wind = additional_wind_results(data)
    print("Calculating whole-fleet wind results and thresholds...", flush=True)
    whole_wind, thresholds = whole_wind_results(data)

    raw.to_csv(output_dir / "chapter4_conventional_raw.csv", index=False)
    controlled.to_csv(output_dir / "chapter4_conventional_controlled.csv", index=False)
    additional_wind.to_csv(output_dir / "chapter4_additional_wind.csv", index=False)
    whole_wind.to_csv(output_dir / "chapter4_whole_wind.csv", index=False)
    thresholds.to_csv(output_dir / "chapter4_wind_helpful_thresholds.csv", index=False)

    create_figures(output_dir, raw, controlled, additional_wind, whole_wind)
    write_summary(output_dir, raw, controlled, additional_wind, whole_wind, thresholds)

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
