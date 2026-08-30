"""Tri-planar saliency overlays and the result figures.

Radiology-standard presentation: axial / sagittal / coronal slices taken through
the maximum-saliency voxel, greyscale CBCT underneath, saliency as a
semi-transparent heatmap with an explicit colorbar and intensity scale.

For a single method, pass a one-entry dict to method_comparison_figure; it
produces the same tri-planar figure with one column.

Colormap is 'inferno' (perceptually uniform, monotonic in lightness). Rainbow
maps are avoided deliberately -- they invent visual edges where the data is
smooth, which is exactly the wrong failure mode for a saliency figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SALIENCY_CMAP = "inferno"


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def max_saliency_voxel(saliency: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(saliency)), saliency.shape))


def _planes(volume: np.ndarray, saliency: np.ndarray, centre: tuple[int, int, int]):
    x, y, z = centre
    return [
        ("axial", volume[:, :, z].T, saliency[:, :, z].T),
        ("coronal", volume[:, y, :].T, saliency[:, y, :].T),
        ("sagittal", volume[x, :, :].T, saliency[x, :, :].T),
    ]


def method_comparison_figure(
    volume,
    maps: dict,
    path,
    title: str = "",
    scores: dict | None = None,
    alpha: float = 0.45,
    threshold: float = 0.2,
) -> None:
    """All methods plus the fused map, side by side, one row per plane.

    `scores` (method -> faithfulness) is printed in each column header so the
    figure carries its own evidence.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = np.squeeze(_to_numpy(volume)).astype(np.float32)
    names = list(maps)
    arrays = {n: np.squeeze(_to_numpy(maps[n])).astype(np.float32) for n in names}

    # Common centre so columns are comparable: driven by the fused map if present.
    driver = arrays.get("fused", arrays[names[0]])
    centre = max_saliency_voxel(driver)

    fig, axes = plt.subplots(3, len(names), figsize=(3.2 * len(names), 9.5), squeeze=False)
    image = None
    for col, name in enumerate(names):
        planes = _planes(vol, arrays[name], centre)
        for row, (plane_name, base, overlay) in enumerate(planes):
            ax = axes[row][col]
            ax.imshow(np.flipud(base), cmap="gray", vmin=-2, vmax=2)
            masked = np.ma.masked_less(np.flipud(overlay), threshold)
            image = ax.imshow(masked, cmap=SALIENCY_CMAP, alpha=alpha, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                header = name.replace("_", " ")
                if scores and name in scores and np.isfinite(scores[name]):
                    header += f"\n{scores[name]:.3f}"
                ax.set_title(header, fontsize=10)
            if col == 0:
                ax.text(-0.08, 0.5, plane_name, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10)

    cbar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("normalised saliency", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def deletion_insertion_curves(results: dict, path, title: str = "",
                              target_is_probability: bool = True) -> None:
    """Deletion and insertion curves for every method on one case.

    `target_is_probability` must match what `deletion_insertion` was given. With
    a millimetre head the y values are STANDARDISED lengths -- unbounded, often
    negative, and under `score="deviation"` never positive. This axis was
    hardcoded to "target-label probability" with `ylim(0, 1)`, which both
    mislabelled the quantity and clipped the whole curve out of the frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, result in results.items():
        fractions = result["fractions"]
        axes[0].plot(fractions, result["deletion_curve"],
                     label=f"{name} (AUC {result['deletion_auc']:.3f})")
        axes[1].plot(fractions, result["insertion_curve"],
                     label=f"{name} (AUC {result['insertion_auc']:.3f})")

    axes[0].set_title("Deletion — lower AUC is better")
    axes[0].set_xlabel("fraction of voxels replaced by baseline")
    axes[1].set_title("Insertion — higher AUC is better")
    axes[1].set_xlabel("fraction of voxels restored")
    ylabel = ("target-label probability" if target_is_probability
              else "target output (standardised, not mm)")
    for ax in axes:
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if target_is_probability:
            ax.set_ylim(0, 1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def agreement_heatmap(matrix: dict, path, key: str = "spearman", title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = matrix["methods"]
    data = np.full((len(names), len(names)), np.nan)
    source = matrix[key] if key == "spearman" else matrix["jaccard"][key]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            value = source.get(f"{a}|{b}", source.get(f"{b}|{a}"))
            if value is not None:
                data[i, j] = value

    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 3, 1.2 * len(names) + 2.5))
    image = ax.imshow(data, cmap="viridis", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] < np.nanmean(data) else "black")
    fig.colorbar(image, ax=ax, label=key)
    ax.set_title(title or f"Inter-method agreement ({key})", fontsize=11)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pareto_curve(rows: list[dict], path, title: str = "") -> None:
    """The money figure: measured compute cost vs faithfulness, with the two
    fixed policies as reference points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    costs = [r["mean_cost_seconds"] for r in rows]
    scores = [r["mean_faithfulness"] for r in rows]
    fractions = [r["ensemble_fraction_actual"] for r in rows]
    n = rows[0]["n_cases"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(costs, scores, "-o", color="#4C72B0", label="adaptive policy (gate swept)")
    for cost, score, fraction in zip(costs, scores, fractions):
        ax.annotate(f"{fraction:.0%}", (cost, score), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)

    ax.scatter([costs[0]], [scores[0]], marker="s", s=90, color="#C44E52", zorder=5,
               label="always cheap (rollout only)")
    ax.scatter([costs[-1]], [scores[-1]], marker="^", s=110, color="#55A868", zorder=5,
               label="always full ensemble")

    ax.set_xlabel("mean measured compute cost per case (seconds)")
    # "lower is better" holds for a PROBABILITY target, where deleting evidence
    # is supposed to drive the score down. On a millimetre head under
    # score="response" it is not founded -- deleting voxels moves a length toward
    # whatever the baseline implies, which may be larger or smaller -- so the
    # axis says which metric it is and leaves the direction to the reader.
    ax.set_ylabel("mean deletion AUC")
    ax.set_title(title or f"Compute vs faithfulness Pareto curve (n = {n} cases)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def randomization_plot(rows: dict[str, list[dict]], path, title: str = "") -> None:
    """Saliency degradation as weights are destroyed, last block backward.

    `rows` maps method name -> the stage rows from model_randomization_check.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not isinstance(rows, dict):
        raise TypeError(f"randomization_plot expects {{method: stages}}, got {type(rows).__name__}")
    for method, stages in rows.items():
        valid = [s for s in stages if "spearman_vs_intact" in s]
        ax.plot(range(len(valid)), [s["spearman_vs_intact"] for s in valid], "-o", label=method)
        if valid:
            ax.set_xticks(range(len(valid)), [s["stage"] for s in valid], rotation=45, ha="right", fontsize=7)

    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_ylabel("Spearman correlation with intact-model map")
    ax.set_xlabel("cumulative weight randomisation (last block first)")
    ax.set_title(title or "Model-randomisation sanity check — a faithful map must degrade", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
