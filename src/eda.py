"""
Exploratory Data Analysis (EDA) for AQI Faisalabad Dataset.

Generates the following analysis plots saved to reports/figures/:
  1. AQI Distribution by category
  2. Seasonal AQI trends (monthly / seasonal boxplots)
  3. Time-of-day and day-of-week heatmap
  4. Pollutant correlation heatmap
  5. Rolling average comparison (3h / 6h / 12h)
  6. AQI change rate distribution
  7. Pollutant concentration time-series
  8. AQI vs PM2.5 scatter plot

Run:
    python -m src.eda
    python -m src.eda --input data/aqi_features.parquet --output reports/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AQI_CATEGORIES, DATA_DIR, LOGS_DIR, POLLUTANTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "eda.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE = "#2196F3 #FF9800 #4CAF50 #F44336 #9C27B0 #00BCD4".split()
AQI_COLORS = [c for _, _, _, c in AQI_CATEGORIES]
AQI_LABELS = [l for _, _, l, _ in AQI_CATEGORIES]
AQI_BOUNDS = [(lo, hi) for lo, hi, _, _ in AQI_CATEGORIES]

sns.set_theme(style="darkgrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "axes.titlecolor": "#ffffff",
    "grid.color": "#2d4a7a",
    "axes.edgecolor": "#2d4a7a",
    "savefig.facecolor": "#1a1a2e",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(input_path: str | None = None) -> pd.DataFrame:
    """Load feature data from a Parquet file."""
    path = Path(input_path) if input_path else DATA_DIR / "aqi_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Run the backfill pipeline first:\n"
            "  python -m src.backfill_pipeline --start 2022-01-01 --end 2024-01-01"
        )
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    log.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def _add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable AQI category column."""
    df = df.copy()

    def _cat(aqi: float) -> str:
        for lo, hi, label, _ in AQI_CATEGORIES:
            if lo <= aqi <= hi:
                return label
        return "Hazardous"

    if "aqi" in df.columns:
        df["aqi_category"] = df["aqi"].apply(_cat)
    return df


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_aqi_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of AQI readings by category."""
    log.info("Plotting AQI distribution by category...")
    df = _add_aqi_category(df)

    cat_counts = df["aqi_category"].value_counts()
    ordered = [l for l in AQI_LABELS if l in cat_counts.index]
    color_map = {l: c for _, _, l, c in AQI_CATEGORIES}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Distribution – Faisalabad", fontsize=16, fontweight="bold")

    # Bar chart
    ax = axes[0]
    bars = ax.bar(
        ordered,
        [cat_counts[c] for c in ordered],
        color=[color_map[c] for c in ordered],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_title("Count by AQI Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Hourly Readings")
    ax.tick_params(axis="x", rotation=30)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{bar.get_height():,}",
            ha="center", va="bottom", fontsize=9,
        )

    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        [cat_counts.get(c, 0) for c in ordered],
        labels=ordered,
        colors=[color_map[c] for c in ordered],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "white", "fontsize": 9},
    )
    ax2.set_title("Proportion by AQI Category")

    plt.tight_layout()
    fig.savefig(out_dir / "01_aqi_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 01_aqi_distribution.png")


def plot_seasonal_trends(df: pd.DataFrame, out_dir: Path) -> None:
    """Monthly and seasonal AQI boxplots."""
    log.info("Plotting seasonal AQI trends...")

    if "month" not in df.columns:
        df = df.copy()
        df["month"] = df["timestamp_utc"].dt.month
    if "season" not in df.columns:
        month_to_season = {12: "Winter", 1: "Winter", 2: "Winter",
                           3: "Spring", 4: "Spring", 5: "Spring",
                           6: "Summer", 7: "Summer", 8: "Summer",
                           9: "Autumn", 10: "Autumn", 11: "Autumn"}
        df = df.copy()
        df["season"] = df["month"].map(month_to_season)

    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Seasonal AQI Trends – Faisalabad", fontsize=16, fontweight="bold")

    # Monthly boxplot
    ax = axes[0]
    monthly_data = [df[df["month"] == m]["aqi"].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(
        monthly_data,
        labels=MONTH_NAMES,
        patch_artist=True,
        medianprops={"color": "#FF9800", "linewidth": 2},
    )
    colors_cycle = plt.cm.coolwarm(np.linspace(0, 1, 12))
    for patch, color in zip(bp["boxes"], colors_cycle):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title("Monthly AQI Distribution")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    ax.axhline(y=150, color="red", linestyle="--", alpha=0.5, label="Unhealthy (150)")
    ax.legend(fontsize=9)

    # Seasonal violin
    ax2 = axes[1]
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_data_plot = df[df["season"].isin(season_order)].copy()
    season_palette = {"Winter": "#64b5f6", "Spring": "#81c784",
                      "Summer": "#ff8a65", "Autumn": "#ffb74d"}
    if not season_data_plot.empty:
        sns.violinplot(
            data=season_data_plot, x="season", y="aqi",
            order=season_order,
            palette=season_palette,
            ax=ax2, inner="box",
        )
    ax2.set_title("Seasonal AQI Distribution")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("AQI")
    ax2.axhline(y=150, color="red", linestyle="--", alpha=0.5, label="Unhealthy (150)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "02_seasonal_trends.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 02_seasonal_trends.png")


def plot_time_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Hour-of-day vs day-of-week AQI heatmap."""
    log.info("Plotting time-of-day heatmap...")

    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = df["timestamp_utc"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp_utc"].dt.dayofweek

    pivot = (
        df.groupby(["day_of_week", "hour"])["aqi"]
        .mean()
        .unstack(level=1)
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [day_names[i] for i in pivot.index]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=False,
        fmt=".0f",
        linewidths=0.3,
        cbar_kws={"label": "Mean AQI", "shrink": 0.8},
    )
    ax.set_title("Mean AQI by Hour and Day of Week – Faisalabad", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Day of Week")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / "03_time_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 03_time_heatmap.png")


def plot_pollutant_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    """Correlation heatmap between pollutants and AQI."""
    log.info("Plotting pollutant correlation heatmap...")

    cols = ["aqi"] + [p for p in POLLUTANTS if p in df.columns]
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
    )
    ax.set_title("Pollutant Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "04_pollutant_correlation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 04_pollutant_correlation.png")


def plot_rolling_averages(df: pd.DataFrame, out_dir: Path) -> None:
    """Compare AQI raw vs 3h/6h/12h rolling averages over a sample period."""
    log.info("Plotting rolling averages comparison...")

    sample = df.sort_values("timestamp_utc").tail(720)  # last 30 days
    ts = sample["timestamp_utc"]

    fig, ax = plt.subplots(figsize=(16, 5))

    if "aqi" in sample.columns:
        ax.plot(ts, sample["aqi"], alpha=0.3, color="#90caf9", linewidth=0.8, label="Raw AQI")
    if "aqi_3h_avg" in sample.columns:
        ax.plot(ts, sample["aqi_3h_avg"], color="#2196F3", linewidth=1.5, label="3h Rolling Avg")
    if "aqi_6h_avg" in sample.columns:
        ax.plot(ts, sample["aqi_6h_avg"], color="#FF9800", linewidth=1.5, label="6h Rolling Avg")
    if "aqi_12h_avg" in sample.columns:
        ax.plot(ts, sample["aqi_12h_avg"], color="#4CAF50", linewidth=2.0, label="12h Rolling Avg")

    ax.axhline(y=150, color="red", linestyle="--", alpha=0.6, label="Unhealthy Threshold (150)")
    ax.set_title("AQI Raw vs Rolling Averages (Last 30 Days)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(out_dir / "05_rolling_averages.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 05_rolling_averages.png")


def plot_aqi_change_rate(df: pd.DataFrame, out_dir: Path) -> None:
    """Distribution of AQI change rate (momentum feature)."""
    log.info("Plotting AQI change rate distribution...")

    if "aqi_change_rate" not in df.columns:
        log.warning("aqi_change_rate column not found, skipping")
        return

    change = df["aqi_change_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    change = change[change.abs() <= 100]  # clip extreme outliers for display

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Change Rate (Momentum Feature)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(change, bins=60, color="#2196F3", edgecolor="white", linewidth=0.3, alpha=0.8)
    ax.axvline(x=0, color="white", linewidth=1.5, linestyle="--")
    ax.set_title("Distribution of AQI % Change")
    ax.set_xlabel("% Change from Previous Hour")
    ax.set_ylabel("Frequency")

    ax2 = axes[1]
    sample = df.sort_values("timestamp_utc").tail(500)
    ax2.plot(sample["timestamp_utc"], sample["aqi_change_rate"].clip(-100, 100),
             color="#FF9800", linewidth=0.8, alpha=0.8)
    ax2.axhline(y=0, color="white", linewidth=1, linestyle="--")
    ax2.set_title("AQI Change Rate Over Time (Last 500 Readings)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("% Change")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(out_dir / "06_aqi_change_rate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 06_aqi_change_rate.png")


def plot_pollutant_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    """Multi-panel pollutant concentration time-series."""
    log.info("Plotting pollutant time-series...")

    avail_pollutants = [p for p in POLLUTANTS if p in df.columns]
    if not avail_pollutants:
        log.warning("No pollutant columns found, skipping time-series plot")
        return

    sample = df.sort_values("timestamp_utc").tail(720)  # last 30 days
    ts = sample["timestamp_utc"]

    n = len(avail_pollutants)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Pollutant Concentrations (Last 30 Days)", fontsize=14, fontweight="bold")

    colors = PALETTE[:n]
    labels = {"pm25": "PM2.5 (μg/m³)", "pm10": "PM10 (μg/m³)",
              "no2": "NO₂ (μg/m³)", "o3": "O₃ (μg/m³)",
              "co": "CO (μg/m³)", "so2": "SO₂ (μg/m³)"}

    for ax, pol, color in zip(axes, avail_pollutants, colors):
        ax.fill_between(ts, sample[pol], alpha=0.4, color=color)
        ax.plot(ts, sample[pol], color=color, linewidth=0.9, label=labels.get(pol, pol))
        ax.set_ylabel(labels.get(pol, pol), fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_dir / "07_pollutant_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 07_pollutant_timeseries.png")


def plot_aqi_vs_pm25(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot of AQI vs PM2.5 coloured by season."""
    log.info("Plotting AQI vs PM2.5 scatter...")

    if "pm25" not in df.columns or "aqi" not in df.columns:
        log.warning("pm25 or aqi column missing, skipping scatter plot")
        return

    df = df.copy()
    if "month" not in df.columns:
        df["month"] = df["timestamp_utc"].dt.month
    month_to_season = {12: "Winter", 1: "Winter", 2: "Winter",
                       3: "Spring", 4: "Spring", 5: "Spring",
                       6: "Summer", 7: "Summer", 8: "Summer",
                       9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month"].map(month_to_season)

    sample = df.dropna(subset=["aqi", "pm25"]).sample(min(5000, len(df)), random_state=42)
    season_palette = {"Winter": "#64b5f6", "Spring": "#81c784",
                      "Summer": "#ff8a65", "Autumn": "#ffb74d"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for season, grp in sample.groupby("season"):
        ax.scatter(
            grp["pm25"], grp["aqi"],
            c=season_palette.get(season, "#888"),
            alpha=0.5, s=15, label=season, edgecolors="none",
        )
    ax.set_title("AQI vs PM2.5 (coloured by season)", fontsize=14, fontweight="bold")
    ax.set_xlabel("PM2.5 Concentration (μg/m³)")
    ax.set_ylabel("AQI")
    ax.legend(title="Season", fontsize=10)

    # Add EPA category background bands
    for (lo, hi), color in zip(AQI_BOUNDS, AQI_COLORS):
        ax.axhspan(lo, hi, alpha=0.08, color=color)

    plt.tight_layout()
    fig.savefig(out_dir / "08_aqi_vs_pm25.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 08_aqi_vs_pm25.png")


def plot_annual_aqi_trend(df: pd.DataFrame, out_dir: Path) -> None:
    """Annual AQI trend with monthly mean ± std band."""
    log.info("Plotting annual AQI trend...")

    df = df.copy()
    df["year_month"] = df["timestamp_utc"].dt.to_period("M")

    monthly = (
        df.groupby("year_month")["aqi"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly[monthly["count"] >= 10]  # skip months with < 10 readings

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(
        monthly["year_month_dt"],
        monthly["mean"] - monthly["std"],
        monthly["mean"] + monthly["std"],
        alpha=0.25, color="#2196F3", label="±1 Std Dev",
    )
    ax.plot(monthly["year_month_dt"], monthly["mean"],
            color="#2196F3", linewidth=2, label="Monthly Mean AQI")
    ax.axhline(y=150, color="red", linestyle="--", alpha=0.5, label="Unhealthy (150)")
    ax.set_title("Annual AQI Trend – Faisalabad", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_dir / "09_annual_aqi_trend.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved 09_annual_aqi_trend.png")


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics to console."""
    cols = ["aqi"] + [p for p in POLLUTANTS if p in df.columns]
    print("\n" + "=" * 65)
    print("  EDA SUMMARY STATISTICS – FAISALABAD AQI DATASET")
    print("=" * 65)
    print(f"  Total rows     : {len(df):,}")
    print(f"  Date range     : {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    print(f"  Cities         : {df['city'].unique().tolist()}")
    print(f"\n  Pollutant & AQI Descriptive Statistics:")
    print(df[cols].describe().round(2).to_string())
    print("\n  AQI Category Breakdown:")
    df2 = _add_aqi_category(df)
    cat_pct = df2["aqi_category"].value_counts(normalize=True) * 100
    for cat, pct in cat_pct.items():
        print(f"    {cat:<40} {pct:>5.1f}%")
    print("=" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eda(input_path: str | None = None, output_dir: str | None = None) -> None:
    log.info("=== EDA started ===")

    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    df = load_data(input_path)

    print_summary_stats(df)

    plot_aqi_distribution(df, out_dir)
    plot_seasonal_trends(df, out_dir)
    plot_time_heatmap(df, out_dir)
    plot_pollutant_correlation(df, out_dir)
    plot_rolling_averages(df, out_dir)
    plot_aqi_change_rate(df, out_dir)
    plot_pollutant_timeseries(df, out_dir)
    plot_aqi_vs_pm25(df, out_dir)
    plot_annual_aqi_trend(df, out_dir)

    log.info("=== EDA complete – figures saved to %s ===", out_dir)
    print(f"\n✅  All EDA figures saved to: {out_dir}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for AQI Faisalabad dataset")
    parser.add_argument("--input",  default=None, help="Path to feature Parquet file")
    parser.add_argument("--output", default=None, help="Output directory for figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_eda(input_path=args.input, output_dir=args.output)
    except Exception:
        log.exception("EDA failed")
        sys.exit(1)
