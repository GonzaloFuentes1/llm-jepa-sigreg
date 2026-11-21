"""Aggregate per-run metrics into results/summary.csv, results/summary.md and a figure.

Every condition is compared with the regular (NTP only) runs that share its padding
mode, label scope and optimizer settings, using a paired t-test over common seeds.

Usage: python scripts/aggregate.py [--results results]
"""

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

PAIRING_KEYS = ("pad_as_eos", "label_scope", "lr", "epochs", "batch_size", "model_id")
# LLM-JEPA paper, Table 3 (Llama-3.2-1B-Instruct, NL-RX-SYNTH, lr 2e-5, lambda 1, k 1).
PAPER = {"regular": 57.29, "LLM-JEPA": 71.46}
EVAL_KEYS = (
    "eot_rate",
    "prefix_match",
    "cos_mean",
    "sigreg_stat",
    "erank_pred",
    "erank_target",
    "erank_joint",
)


def load_runs(results: Path, pilot: bool = False) -> dict[str, list[dict]]:
    """Runs grouped by tag; pilot runs (tags starting with "pilot") are kept apart."""
    runs = defaultdict(list)
    for path in sorted(results.glob("runs/*/s*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["tag"].startswith("pilot") != pilot:
            continue
        predictions = path.with_name(path.stem + ".predictions.jsonl.gz")
        with gzip.open(predictions, "rt", encoding="utf-8") as f:
            preds = [json.loads(line) for line in f]
        # Generation starts with the exact reference (it may continue after it).
        run["eval"]["prefix_match"] = statistics.fmean(
            p["prediction"].strip().startswith(p["reference"]) for p in preds
        )
        runs[run["tag"]].append(run)
    return runs


def write_pilot_table(pilot: dict[str, list[dict]], results: Path) -> None:
    """Validation (held-out training examples) results of the beta selection pilot."""
    lines = [
        "| beta | split | exact match (%) | EOT rate | SIGReg | cos | erank pred / target |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = sorted((r for runs in pilot.values() for r in runs), key=lambda r: r["config"]["beta"])
    for r in rows:
        e = r["eval"]
        lines.append(
            f"| {r['config']['beta']:g} | {e['split']} | {100 * e['exact_match']:.2f} "
            f"| {e['eot_rate']:.3f} | {e['sigreg_stat']:.1f} | {e['cos_mean']:.3f} "
            f"| {e['erank_pred']:.1f} / {e['erank_target']:.1f} |"
        )
    _write(results / "pilot_beta.md", "\n".join(lines) + "\n")


def _mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mean, std


def find_reference(tag: str, runs: dict[str, list[dict]], **overrides) -> str | None:
    """The other condition that matches tag on the pairing keys, lambda and beta,
    except for the values given in ``overrides``."""
    config = runs[tag][0]["config"]
    target = {k: config[k] for k in (*PAIRING_KEYS, "lbd", "beta")} | overrides
    for other, other_runs in runs.items():
        c = other_runs[0]["config"]
        if other != tag and all(c[k] == v for k, v in target.items()):
            return other
    return None


def paired_test(cond: list[dict], base: list[dict]):
    by_seed = {r["config"]["seed"]: r["eval"]["exact_match"] for r in base}
    pairs = [
        (r["eval"]["exact_match"], by_seed[r["config"]["seed"]])
        for r in cond
        if r["config"]["seed"] in by_seed
    ]
    if len(pairs) < 2:
        return None, None, None
    a, b = zip(*pairs, strict=True)
    diff = statistics.fmean(a) - statistics.fmean(b)
    two_sided = stats.ttest_rel(a, b).pvalue
    greater = stats.ttest_rel(a, b, alternative="greater").pvalue
    return diff, two_sided, greater


def summarize(runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for tag, tag_runs in runs.items():
        config = tag_runs[0]["config"]
        em_mean, em_std = _mean_std([100 * r["eval"]["exact_match"] for r in tag_runs])
        row = {
            "condition": tag,
            "n_seeds": len(tag_runs),
            "seeds": " ".join(str(r["config"]["seed"]) for r in tag_runs),
            "lambda": config["lbd"],
            "beta": config["beta"],
            "k": config["predictors"],
            "pad_as_eos": config["pad_as_eos"],
            "label_scope": config["label_scope"],
            "epochs": config["epochs"],
            "em_mean": em_mean,
            "em_std": em_std,
        }
        for key in EVAL_KEYS:
            row[key] = statistics.fmean(r["eval"][key] for r in tag_runs)
        row["train_minutes"] = statistics.fmean(r["train"]["train_seconds"] for r in tag_runs) / 60
        row["eval_minutes"] = statistics.fmean(r["eval"]["eval_seconds"] for r in tag_runs) / 60
        baseline = find_reference(tag, runs, lbd=0.0, beta=0.0)
        diff, p2, p1 = paired_test(tag_runs, runs[baseline]) if baseline else (None,) * 3
        row["baseline"] = baseline or ""
        row["delta_vs_baseline"] = None if diff is None else 100 * diff
        row["p_paired_two_sided"] = p2
        row["p_paired_greater"] = p1
        # SIGReg runs are also compared with the same lambda without SIGReg, and pad = eos
        # runs with the same condition using the reserved pad token.
        for name, needed, overrides in (
            ("no_sigreg", config["beta"] > 0, {"beta": 0.0}),
            ("dedicated_pad", config["pad_as_eos"], {"pad_as_eos": False}),
        ):
            reference = find_reference(tag, runs, **overrides) if needed else None
            diff, p2, _ = paired_test(tag_runs, runs[reference]) if reference else (None,) * 3
            row[f"{name}_reference"] = reference or ""
            row[f"delta_vs_{name}"] = None if diff is None else 100 * diff
            row[f"p_vs_{name}_two_sided"] = p2
        rows.append(row)
    rows.sort(
        key=lambda r: (r["epochs"] > 0, r["pad_as_eos"], r["label_scope"], r["beta"], r["lambda"])
    )
    return rows


def _fmt(value, digits=2):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if value != value:  # nan
            return ""
        return f"{value:.{digits}g}" if abs(value) < 1e-3 else f"{value:.{digits}f}"
    return str(value)


def write_tables(rows: list[dict], results: Path) -> None:
    with open(results / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    header = (
        "| condition | n | exact match (%) | delta vs regular | p (paired, two-sided) "
        "| EOT rate | prefix match | test SIGReg | erank pred / target |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['n_seeds']} | {_em(r)} | {_delta(r['delta_vs_baseline'])} "
            f"| {_p(r['p_paired_two_sided'])} | {r['eot_rate']:.3f} | {r['prefix_match']:.3f} "
            f"| {r['sigreg_stat']:.1f} | {r['erank_pred']:.1f} / {r['erank_target']:.1f} |"
        )
    by_tag = {r["condition"]: r for r in rows}

    def comparison(name: str, label: str) -> list[str]:
        table = [
            f"| condition | exact match (%) | {label} | reference (%) "
            "| delta | p (paired, two-sided) |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            if r[f"{name}_reference"]:
                ref = by_tag[r[f"{name}_reference"]]
                table.append(
                    f"| {r['condition']} | {_em(r)} | {ref['condition']} | {_em(ref)} "
                    f"| {_delta(r[f'delta_vs_{name}'])} | {_p(r[f'p_vs_{name}_two_sided'])} |"
                )
        return table

    sigreg = comparison("no_sigreg", "reference without SIGReg")
    pad = comparison("dedicated_pad", "reference with reserved pad")
    sensitivity = [
        "| series | lambda = 0.5 | lambda = 1 | lambda = 2 | max - min of means |",
        "|---|---|---|---|---|",
    ]
    series = defaultdict(dict)
    for r in rows:
        if r["lambda"] > 0 and r["epochs"] > 0 and r["label_scope"] == "all":
            name = "LLM-JEPA" if r["beta"] == 0 else f"LLM-JEPA + SIGReg (beta={r['beta']:g})"
            series[name + (", pad = eos" if r["pad_as_eos"] else "")][r["lambda"]] = r
    for name, by_lambda in series.items():
        if len(by_lambda) < 2:
            continue
        cells = [_em(by_lambda[lam]) if lam in by_lambda else "" for lam in (0.5, 1.0, 2.0)]
        means = [r["em_mean"] for r in by_lambda.values()]
        sensitivity.append(f"| {name} | {' | '.join(cells)} | {max(means) - min(means):.2f} |")
    text = (
        header
        + "\n".join(lines)
        + "\n\nSIGReg against the same lambda without SIGReg (paired by seed):\n\n"
        + "\n".join(sigreg)
        + "\n\nOfficial pad = eos against the reserved pad token (paired by seed):\n\n"
        + "\n".join(pad)
        + "\n\nSensitivity to lambda (exact match %, mean ± std over seeds):\n\n"
        + "\n".join(sensitivity)
        + "\n"
    )
    _write(results / "summary.md", text)


def _write(path: Path, text: str) -> None:
    """UTF-8 with LF line endings on every platform."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _em(row: dict) -> str:
    if row["n_seeds"] > 1:
        return f"{row['em_mean']:.2f} ± {row['em_std']:.2f}"
    return f"{row['em_mean']:.2f}"


def _delta(delta: float | None) -> str:
    return "" if delta is None else f"{delta:+.2f}"


def _p(p: float | None) -> str:
    if p is None:
        return ""
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def plot(runs: dict[str, list[dict]], rows: list[dict], results: Path) -> None:
    trained = [r for r in rows if r["epochs"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for x, row in enumerate(trained):
        values = [100 * r["eval"]["exact_match"] for r in runs[row["condition"]]]
        color = (
            "tab:red" if row["pad_as_eos"] else ("tab:blue" if row["beta"] == 0 else "tab:green")
        )
        ax.bar(x, row["em_mean"], color=color, alpha=0.35)
        ax.scatter([x] * len(values), values, color=color, s=14, zorder=3)
    ax.set_xticks(range(len(trained)), [r["condition"] for r in trained], rotation=60, ha="right")
    ax.set_ylabel("exact match on synth_test (%)")
    ax.set_title("Per-seed exact match (bars: mean)")
    lo = min(100 * r["eval"]["exact_match"] for row in trained for r in runs[row["condition"]])
    ax.set_ylim(max(0, lo - 10), 100)
    for (name, value), style in zip(PAPER.items(), (":", "--"), strict=True):
        ax.axhline(value, color="black", ls=style, lw=1, label=f"paper, {name} ({value})")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    series = defaultdict(list)
    for row in trained:
        if row["pad_as_eos"] or row["lambda"] == 0 or row["label_scope"] != "all":
            continue
        series[row["beta"]].append((row["lambda"], row["em_mean"], row["em_std"]))
    for beta, points in sorted(series.items()):
        if len(points) < 2:
            continue
        points.sort()
        lam, mean, std = zip(*points, strict=True)
        label = "LLM-JEPA" if beta == 0 else f"LLM-JEPA + SIGReg (beta={beta:g})"
        ax.errorbar(lam, mean, yerr=std, marker="o", capsize=3, label=label)
    base = [r for r in trained if r["lambda"] == 0 and r["beta"] == 0 and not r["pad_as_eos"]]
    if base:
        ax.axhline(base[0]["em_mean"], color="gray", ls="--", label="regular (NTP)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("lambda")
    ax.set_ylabel("exact match (%)")
    ax.set_title("Sensitivity to lambda (mean ± std over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "figure.png", dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    results = Path(args.results)
    runs = load_runs(results)
    rows = summarize(runs)
    write_tables(rows, results)
    plot(runs, rows, results)
    pilot = load_runs(results, pilot=True)
    if pilot:
        write_pilot_table(pilot, results)
    print((results / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
