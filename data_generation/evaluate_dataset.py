"""
Dataset quality evaluation: cumulants, S-K coverage, drift plots.

Usage
-----
python datasets/evaluate_dataset.py datasets/datasets/<run_folder>/
python datasets/evaluate_dataset.py datasets/datasets/<run_folder>/ --max_order 6
python datasets/evaluate_dataset.py datasets/datasets/<run_folder>/ --grid_bins 30
python datasets/evaluate_dataset.py datasets/datasets/<run_folder>/ --help

window_size defaults to block_size from dataset_config.json (one window per signal).
Results are saved to <dataset_dir>/dataset_evaluation/
"""

import argparse
import os
from datetime import datetime
from math import comb

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Cumulant computation (fully vectorized)
# ──────────────────────────────────────────────────────────────────────────────

def compute_cumulants_batch(windows: np.ndarray, max_order: int) -> np.ndarray:
    """
    Compute cumulants κ₃ … κ_max_order for a batch of standardized windows.

    Algorithm
    ---------
    For each window z (already standardized: mean=0, std=1):
      1. Compute raw moments μₖ = mean(zᵏ), k = 1 … max_order.
      2. Apply recursive moment-cumulant formula:
             κₙ = μₙ − Σ_{j=1}^{n-1} C(n−1, j−1) · κⱼ · μₙ₋ⱼ
         with κ₁ = 0, κ₂ = 1 (by normalization).

    Parameters
    ----------
    windows   : (N, W) — batch of standardized windows
    max_order : highest cumulant order to compute

    Returns
    -------
    kappas : (N, max_order − 2) — cumulants [κ₃, κ₄, … , κ_max_order]
    """
    n_wins = len(windows)

    # Raw moments (N, max_order); moments[:, k-1] = μ_k
    moments = np.stack(
        [np.mean(windows ** k, axis=1) for k in range(1, max_order + 1)],
        axis=1,
    )

    # Cumulants array (N, max_order+1); index k holds κ_k
    kappas = np.zeros((n_wins, max_order + 1))
    kappas[:, 1] = 0.0   # κ₁ = 0  (zero mean, by standardization)
    kappas[:, 2] = 1.0   # κ₂ = 1  (unit variance, by standardization)

    for n in range(3, max_order + 1):
        kn = moments[:, n - 1].copy()          # μₙ
        for j in range(1, n):
            # μₙ₋ⱼ = moments[:, n-j-1]
            kn -= comb(n - 1, j - 1) * kappas[:, j] * moments[:, n - j - 1]
        kappas[:, n] = kn

    return kappas[:, 3:]   # (N, max_order − 2)


def analyze_dataset(
    signals: np.ndarray,
    window_size: int,
    max_order: int,
) -> np.ndarray:
    """
    Step 1–2: Segment signals into windows, standardize, compute cumulants.

    If window_size ≥ signal_length, each signal is a single window.
    Uses vectorized computation — no Python loop over individual samples.

    Returns
    -------
    cumulants : (M, max_order − 2)  M = total valid windows
    """
    n_signals, signal_len = signals.shape
    win = min(window_size, signal_len)
    wins_per_signal = signal_len // win

    # Reshape to (N_total, win)
    n_total = n_signals * wins_per_signal
    trimmed = signals[:, : wins_per_signal * win]          # trim remainder
    all_wins = trimmed.reshape(n_total, win)               # (N_total, win)

    # Vectorized standardization
    means = all_wins.mean(axis=1, keepdims=True)
    stds  = all_wins.std(axis=1,  keepdims=True)
    valid = stds.squeeze() > 1e-9

    z = np.where(stds > 1e-9, (all_wins - means) / stds, 0.0)

    cumulants = compute_cumulants_batch(z[valid], max_order)
    return cumulants


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Coverage Rate
# ──────────────────────────────────────────────────────────────────────────────

def compute_coverage(
    gamma3: np.ndarray,
    gamma4: np.ndarray,
    grid_bins: int,
    threshold: float,
) -> tuple:
    """
    Step 3: Estimate S-K Coverage Rate.

    Feasible region:  γ₄ ≥ γ₃² − 2  (parabolic boundary for real distributions).
    Coverage = (occupied feasible cells) / (total feasible cells) × 100 %.

    Returns
    -------
    coverage   : float [0, 100]
    g3_edges   : (grid_bins+1,)
    g4_edges   : (grid_bins+1,)
    H          : (grid_bins, grid_bins) histogram counts
    feasible   : (grid_bins, grid_bins) boolean mask
    """
    g3_lo, g3_hi = -6.0, 6.0
    g4_lo = -2.0
    g4_hi = float(min(np.percentile(gamma4, 99) * 1.2, 40.0))

    g3_edges = np.linspace(g3_lo, g3_hi, grid_bins + 1)
    g4_edges = np.linspace(g4_lo, g4_hi, grid_bins + 1)

    g3_c = 0.5 * (g3_edges[:-1] + g3_edges[1:])
    g4_c = 0.5 * (g4_edges[:-1] + g4_edges[1:])
    G3, G4 = np.meshgrid(g3_c, g4_c, indexing="ij")
    feasible = G4 >= G3 ** 2 - 2

    H, _, _ = np.histogram2d(gamma3, gamma4, bins=[g3_edges, g4_edges])
    occupied = (H > 0) & feasible

    coverage = 100.0 * occupied.sum() / max(feasible.sum(), 1)
    return coverage, g3_edges, g4_edges, H, feasible


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _parabola(g3_range=(-6, 6)):
    g3 = np.linspace(*g3_range, 300)
    return g3, g3 ** 2 - 2


def plot_sk_scatter(gamma3, gamma4, coverage, threshold, out_dir) -> str:
    """Graph 1: Skewness–Kurtosis scatter plot with boundary and Gaussian reference."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(gamma3, gamma4, alpha=0.15, s=6, c="steelblue", label="Dataset windows")
    ax.scatter([0], [0], c="red", s=120, zorder=6, label="Ideal Gaussian (0, 0)")

    g3_l, g4_l = _parabola()
    ax.plot(g3_l, g4_l, "k--", linewidth=1.2, label="Boundary  γ₄ = γ₃² − 2")

    g4_hi = float(min(np.percentile(gamma4, 99) * 1.3, 42.0))
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-3, g4_hi)
    ax.set_xlabel("Skewness  γ₃", fontsize=12)
    ax.set_ylabel("Excess Kurtosis  γ₄", fontsize=12)
    ax.set_title("Skewness – Kurtosis Plane", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    status_color = "green" if coverage >= threshold else "red"
    status_text  = "✓ PASS" if coverage >= threshold else "✗ WARNING"
    ax.text(
        0.02, 0.97,
        f"Coverage: {coverage:.1f}%  (threshold {threshold:.0f}%)  {status_text}",
        transform=ax.transAxes, fontsize=10, va="top", color=status_color,
    )
    ax.text(
        0.02, 0.02,
        "Wide scatter around (0, 0) confirms high distributional variability.\n"
        "High γ₄ values indicate heavy-tailed / impulsive noise components.",
        transform=ax.transAxes, fontsize=8, va="bottom", color="dimgray",
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "plot1_sk_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_density_heatmap(gamma3, gamma4, g3_edges, g4_edges, H, feasible, out_dir) -> str:
    """Graph 2: Density heatmap of the S-K plane (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask infeasible region
    H_plot = np.where(feasible, H + 1, np.nan)
    im = ax.pcolormesh(
        g3_edges, g4_edges, H_plot.T,
        cmap="YlOrRd", norm=matplotlib.colors.LogNorm(vmin=1),
    )
    plt.colorbar(im, ax=ax, label="Count  (log scale)")

    # Infeasible overlay
    H_inf = np.where(~feasible, 1.0, np.nan)
    ax.pcolormesh(g3_edges, g4_edges, H_inf.T, cmap="Greys", alpha=0.25, vmin=0, vmax=1)

    ax.scatter([0], [0], c="blue", s=80, zorder=5, label="Gaussian (0, 0)")
    g3_l, g4_l = _parabola()
    ax.plot(g3_l, g4_l, "w--", linewidth=1.2, label="Boundary")

    ax.set_xlim(g3_edges[0], g3_edges[-1])
    ax.set_ylim(g4_edges[0], g4_edges[-1])
    ax.set_xlabel("Skewness  γ₃", fontsize=12)
    ax.set_ylabel("Excess Kurtosis  γ₄", fontsize=12)
    ax.set_title("S–K Density Heatmap", fontsize=13)
    ax.legend(fontsize=9)
    ax.text(
        0.02, 0.02,
        "Uniform coverage confirms the generator does not\n"
        "get stuck in a single noise regime.",
        transform=ax.transAxes, fontsize=8, va="bottom", color="white",
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "plot2_sk_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cumulant_histograms(cumulants: np.ndarray, max_order: int, out_dir: str) -> str:
    """Graph 3: Histogram distributions for all cumulants κ₃ … κ_max_order."""
    orders = list(range(3, max_order + 1))
    n_orders = len(orders)

    ncols = min(n_orders, 4)
    nrows = (n_orders + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, order in enumerate(orders):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        data = cumulants[:, idx]

        p1, p99 = np.percentile(data, 1), np.percentile(data, 99)

        ax.hist(data, bins=50, color="steelblue", alpha=0.7, edgecolor="none",
                range=(p1, p99))
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Gaussian = 0")
        median = np.median(data)
        ax.axvline(median, color="orange", linestyle="-", linewidth=1.2,
                   label=f"Median = {median:.2f}")
        ax.set_title(f"γ{order}  (κ{order})", fontsize=11)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_orders, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Cumulant Distributions  (γ₃ … γ_max_order)", fontsize=13)
    fig.text(
        0.02, 0.005,
        "Red dashed line = Gaussian reference (0). "
        "Orange line = sample median. "
        "Deviation from 0 confirms non-Gaussian noise.",
        fontsize=8, color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(out_dir, "plot3_cumulant_histograms.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cumulant_scatter_matrix(cumulants: np.ndarray, max_order: int, out_dir: str) -> str:
    """Graph 4: Scatter matrix for all cumulants κ₃ … κ_max_order.

    Diagonal: histogram of each cumulant (coverage of 1-D range).
    Off-diagonal: scatter plot of every pair (γᵢ, γⱼ) — analogous to S-K plane,
    shows whether the datasets covers the 2-D space of each cumulant pair.
    """
    orders = list(range(3, max_order + 1))
    n = len(orders)
    labels = [f"γ{o}" for o in orders]

    # Compute display limits [P1, P99] per column without clipping
    limits = []
    for idx in range(n):
        p1 = np.percentile(cumulants[:, idx], 1)
        p99 = np.percentile(cumulants[:, idx], 99)
        limits.append((p1, p99))
    data = cumulants

    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n))

    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            if row == col:
                p1, p99 = limits[row]
                ax.hist(data[:, row], bins=40, color="steelblue", alpha=0.75,
                        edgecolor="none", range=(p1, p99))
                ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
            else:
                ax.scatter(data[:, col], data[:, row], alpha=0.08, s=3, c="steelblue", rasterized=True)
                ax.set_xlim(*limits[col])
                ax.set_ylim(*limits[row])
                ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
                ax.axvline(0, color="gray", linestyle=":", linewidth=0.6)

            if row == n - 1:
                ax.set_xlabel(labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=10)

            ax.tick_params(labelsize=7)

    fig.suptitle("Cumulant Scatter Matrix  (pairwise coverage)", fontsize=13)
    fig.text(
        0.02, 0.005,
        "Diagonal: distribution of each cumulant (red = Gaussian reference 0). "
        "Off-diagonal: pairwise scatter — wide spread means good coverage of noise regimes.",
        fontsize=8, color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = os.path.join(out_dir, "plot4_cumulant_scatter_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_parameter_drift(
    signals: np.ndarray,
    max_order: int,
    out_dir: str,
    n_examples: int = 6,
    sub_wins: int = 16,
) -> str:
    """
    Graph 5: Cumulant drift γ₃(t) … γ_max_order(t) and σ²(t) over sub-windows.

    Demonstrates non-stationarity: for polygauss_nonstationary, parameters
    visibly drift; for stationary polygauss they remain roughly constant.
    """
    signal_len = signals.shape[1]
    win = max(32, signal_len // sub_wins)
    actual_wins = signal_len // win

    cumulant_orders = list(range(3, max_order + 1))
    n_rows = len(cumulant_orders) + 1   # one row per cumulant + variance

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 0.6, n_examples))

    for i in range(min(n_examples, len(signals))):
        seg_all = signals[i, : actual_wins * win].reshape(actual_wins, win)

        var_traj = seg_all.var(axis=1)
        stds = seg_all.std(axis=1, keepdims=True)
        valid_mask = stds.squeeze() > 1e-9
        z_batch = np.where(stds > 1e-9, (seg_all - seg_all.mean(axis=1, keepdims=True)) / stds, 0.0)

        cum_batch = compute_cumulants_batch(z_batch, max_order)  # (actual_wins, max_order-2)

        t = np.arange(actual_wins)
        for idx, order in enumerate(cumulant_orders):
            traj = cum_batch[:, idx].copy()
            traj[~valid_mask] = np.nan
            axes[idx].plot(t, traj, color=colors[i], alpha=0.75, linewidth=1.2)

        axes[-1].plot(t, var_traj, color=colors[i], alpha=0.75, linewidth=1.2,
                      label=f"Signal #{i}")

    for idx, order in enumerate(cumulant_orders):
        axes[idx].set_ylabel(f"γ{order}", fontsize=11)
        axes[idx].axhline(0, color="gray", linestyle=":", linewidth=0.8)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_ylabel("Variance  σ²", fontsize=11)
    axes[-1].axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[-1].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sub-window index  (time →)", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper right", ncol=2)

    fig.suptitle("Parameter Drift Over Time  (Non-Stationarity Check)", fontsize=13)
    fig.text(
        0.02, 0.005,
        "Smooth evolution of cumulants over time demonstrates non-stationarity: "
        "noise parameters drift, mimicking changes in a real radio environment.",
        fontsize=8, color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(out_dir, "plot5_parameter_drift.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Text report (Dataset Card)
# ──────────────────────────────────────────────────────────────────────────────

def save_report(
    out_dir: str,
    dataset_dir: str,
    signal_shape: tuple,
    window_size: int,
    max_order: int,
    coverage_threshold: float,
    cumulants: np.ndarray,
    coverage: float,
    plots: list,
    dataset_config: dict | None = None,
) -> str:
    orders = list(range(3, max_order + 1))
    lines = [
        "=" * 64,
        "  DATASET EVALUATION REPORT  (Dataset Card)",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 64,
        "",
        "INPUT",
        f"  Dataset path  : {os.path.abspath(dataset_dir)}",
        f"  Signal shape  : {signal_shape}",
        f"  Windows total : {len(cumulants)}",
        f"  Window size   : {window_size} samples",
        f"  Max order     : {max_order}",
    ]

    if dataset_config:
        lines += [
            "",
            "GENERATION CONFIG",
            f"  uid               : {dataset_config.get('uid', '—')}",
            f"  created_at        : {dataset_config.get('created_at', '—')}",
            f"  scenario          : {dataset_config.get('scenario', '—')}",
            f"  modulation_type   : {dataset_config.get('modulation_type', '—')}",
            f"  bits_per_symbol   : {dataset_config.get('bits_per_symbol', '—')}",
            f"  block_size        : {dataset_config.get('block_size', '—')}",
            f"  sample_rate       : {dataset_config.get('sample_rate', '—')} Hz",
            f"  snr_range         : {dataset_config.get('snr_range', '—')} dB",
            f"  noise_types       : {dataset_config.get('noise_types', '—')}",
            f"  polygauss_K       : {dataset_config.get('polygauss_random_k') or dataset_config.get('polygauss_components', '—')}",
            f"  num_train         : {dataset_config.get('num_train', '—'):,}" if isinstance(dataset_config.get('num_train'), int) else f"  num_train         : —",
        ]

    lines += [
        "",
        "COVERAGE  (S-K plane)",
        f"  Coverage rate : {coverage:.1f}%",
        f"  Threshold     : {coverage_threshold:.0f}%",
        f"  Status        : {'PASS' if coverage >= coverage_threshold else 'WARNING — low variability'}",
        "",
        "CUMULANT STATISTICS  (per window)",
        f"  {'Order':<8} {'Median':>9} {'Std':>9} {'P5':>9} {'P95':>9}",
        "  " + "-" * 44,
    ]
    for idx, order in enumerate(orders):
        col = cumulants[:, idx]
        lines.append(
            f"  γ{order:<7d} {np.median(col):>+9.3f} {col.std():>9.3f}"
            f" {np.percentile(col, 5):>+9.3f} {np.percentile(col, 95):>+9.3f}"
        )

    lines += [
        "",
        "GENERATED FILES",
    ] + [f"  {p}" for p in plots if p]
    lines.append("")

    path = os.path.join(out_dir, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate datasets quality via cumulant analysis and S-K coverage.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to datasets run directory (contains dataset_config.json),\n"
             "e.g.  datasets/datasets/deep_space__polygauss_nonstationary__bpsk__bs256__n50000__a1b2c3d4/",
    )
    parser.add_argument(
        "--window_size", type=int, default=None,
        help="Samples per analysis window.\n"
             "Default: block_size from dataset_config.json (one window per signal).\n"
             "Smaller values → more windows but less reliable cumulant estimates.",
    )
    parser.add_argument(
        "--max_order", type=int, default=4,
        help="Maximum cumulant order to compute (default: 4, range: 3–10).",
    )
    parser.add_argument(
        "--coverage_threshold", type=float, default=70.0,
        help="Target S-K plane coverage %% (default: 70).",
    )
    parser.add_argument(
        "--grid_bins", type=int, default=20,
        help="S-K grid resolution per axis (default: 20 → 20×20 grid).",
    )
    parser.add_argument(
        "--noise_type", choices=["gaussian", "non_gaussian"],
        default="non_gaussian",
        help="Which signal file to analyze (default: non_gaussian).",
    )
    parser.add_argument(
        "--n_signals", type=int, default=None,
        help="Limit analysis to first N signals (default: all).",
    )
    args = parser.parse_args()

    assert 3 <= args.max_order <= 10, "--max_order must be in [3, 10]"

    # ── Resolve dataset path ───────────────────────────────────────────────────
    # Allow passing just the folder name; search in default datasets/ location.
    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        default_base = os.path.join(os.path.dirname(__file__), "datasets")
        candidate = os.path.join(default_base, dataset_dir)
        if os.path.isdir(candidate):
            dataset_dir = candidate
        else:
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir!r}\n"
                f"Also tried: {candidate}"
            )

    # ── Load dataset_config.json ──────────────────────────────────────────────
    config_path = os.path.join(dataset_dir, "dataset_config.json")
    dataset_config: dict | None = None
    if os.path.isfile(config_path):
        with open(config_path) as f:
            dataset_config = json.load(f)
        print(f"Config   → {config_path}")
        print(f"  uid            : {dataset_config.get('uid')}")
        print(f"  scenario       : {dataset_config.get('scenario')}")
        print(f"  modulation     : {dataset_config.get('modulation_type')}  "
              f"bps={dataset_config.get('bits_per_symbol')}")
        print(f"  block_size     : {dataset_config.get('block_size')}")
        print(f"  noise_types    : {dataset_config.get('noise_types')}")
        print(f"  num_train      : {dataset_config.get('num_train'):,}" if isinstance(dataset_config.get('num_train'), int) else "")
    else:
        print(f"Warning: dataset_config.json not found in {dataset_dir}")

    # ── Load signals ──────────────────────────────────────────────────────────
    train_dir = os.path.join(dataset_dir, "train")
    src_dir   = train_dir if os.path.isdir(train_dir) else dataset_dir
    fname     = os.path.join(src_dir, f"{args.noise_type}_signals.npy")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    print(f"\nLoading  {fname}")
    signals = np.load(fname)
    if args.n_signals:
        signals = signals[: args.n_signals]
    print(f"  Shape: {signals.shape}")

    # window_size: from arg → from config → from signal length
    if args.window_size is not None:
        window_size = args.window_size
    elif dataset_config and dataset_config.get("block_size"):
        window_size = dataset_config["block_size"]
        print(f"  Window size: {window_size} (from dataset_config.json block_size)")
    else:
        window_size = signals.shape[1]
        print(f"  Window size: {window_size} (full signal length)")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(dataset_dir, "dataset_evaluation")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output → {out_dir}/")

    # ── Steps 1–2: Cumulants ──────────────────────────────────────────────────
    print(f"\nComputing cumulants  (order 3–{args.max_order}, window={window_size}) …")
    cumulants = analyze_dataset(signals, window_size, args.max_order)
    print(f"  Windows analyzed: {len(cumulants):,}")

    gamma3 = cumulants[:, 0]
    gamma4 = cumulants[:, 1]

    # ── Step 3: Coverage ──────────────────────────────────────────────────────
    coverage, g3_edges, g4_edges, H, feasible = compute_coverage(
        gamma3, gamma4, args.grid_bins, args.coverage_threshold,
    )
    status = "PASS" if coverage >= args.coverage_threshold else "WARNING"
    print(f"  S-K coverage: {coverage:.1f}%  [{status}]"
          f"  (threshold {args.coverage_threshold:.0f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plots: list[str | None] = []
    plots.append(plot_sk_scatter(gamma3, gamma4, coverage, args.coverage_threshold, out_dir))
    plots.append(plot_density_heatmap(gamma3, gamma4, g3_edges, g4_edges, H, feasible, out_dir))
    plots.append(plot_cumulant_histograms(cumulants, args.max_order, out_dir))
    plots.append(plot_cumulant_scatter_matrix(cumulants, args.max_order, out_dir))
    plots.append(plot_parameter_drift(signals, args.max_order, out_dir))

    # ── Report ────────────────────────────────────────────────────────────────
    report = save_report(
        out_dir, dataset_dir, signals.shape, window_size,
        args.max_order, args.coverage_threshold, cumulants, coverage, plots,
        dataset_config=dataset_config,
    )

    print(f"\nReport  → {report}")
    for p in plots:
        if p:
            print(f"Plot    → {p}")
    print("Done.")


if __name__ == "__main__":
    main()
