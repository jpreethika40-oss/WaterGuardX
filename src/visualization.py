"""Visualization utilities for EDA and training analysis."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from config import SENSOR_COLS, PRESSURE_SENSORS, FLOW_SENSORS, LEVEL_SENSORS, TARGET_COL, TIMESTAMP_COL

sns.set_theme(style='whitegrid', palette='muted')
FIGDIR = Path('figures')


def _save(fig, subdir: str, name: str):
    path = FIGDIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figures/{subdir}/{name}")


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ['Normal', 'Anomaly']
    colors = ['#4C72B0', '#DD8452']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, counts.values, color=colors, edgecolor='white', width=0.5)
    axes[0].set_title('Class Distribution (Count)')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontsize=10)
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (%)')
    fig.suptitle('Target Class Distribution', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '01_class_distribution.png')


def plot_sensor_distributions(df: pd.DataFrame):
    n = len(SENSOR_COLS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(SENSOR_COLS):
        normal  = df[df[TARGET_COL] == 0][col]
        anomaly = df[df[TARGET_COL] == 1][col]
        axes[i].hist(normal,  bins=60, alpha=0.6, label='Normal',  color='#4C72B0', density=True)
        axes[i].hist(anomaly, bins=60, alpha=0.6, label='Anomaly', color='#DD8452', density=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].legend(fontsize=7)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Sensor Distributions: Normal vs Anomaly', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '02_sensor_distributions.png')


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[SENSOR_COLS].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, annot_kws={'size': 8},
                linewidths=0.5, square=True)
    ax.set_title('Sensor Correlation Matrix', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '03_correlation_heatmap.png')


def plot_timeseries_overview(df: pd.DataFrame):
    """Plot all sensors over time with anomaly regions shaded."""
    sensor_groups = [
        ('Pressure Sensors (m)',  PRESSURE_SENSORS[:4]),
        ('Pressure Sensors (m)',  PRESSURE_SENSORS[4:]),
        ('Flow Sensors (m³/h)',   FLOW_SENSORS),
        ('Level Sensor (m)',      LEVEL_SENSORS),
    ]
    fig, axes = plt.subplots(len(sensor_groups), 1, figsize=(18, 14), sharex=True)
    anomaly_mask = df[TARGET_COL] == 1
    for ax, (title, sensors) in zip(axes, sensor_groups):
        for col in sensors:
            ax.plot(df[TIMESTAMP_COL], df[col], lw=0.4, alpha=0.8, label=col)
        # Shade anomaly regions
        in_anom = False
        start_t = None
        for _, row in df[[TIMESTAMP_COL, TARGET_COL]].iterrows():
            if row[TARGET_COL] == 1 and not in_anom:
                in_anom = True
                start_t = row[TIMESTAMP_COL]
            elif row[TARGET_COL] == 0 and in_anom:
                in_anom = False
                ax.axvspan(start_t, row[TIMESTAMP_COL], alpha=0.15, color='red', label='_nolegend_')
        if in_anom:
            ax.axvspan(start_t, df[TIMESTAMP_COL].iloc[-1], alpha=0.15, color='red')
        ax.set_ylabel(title, fontsize=8)
        ax.legend(fontsize=7, loc='upper right', ncol=4)
        ax.tick_params(labelsize=7)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    fig.suptitle('Sensor Time Series (red = anomaly periods)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '04_timeseries_overview.png')


def plot_anomaly_timeline(df: pd.DataFrame):
    """Show when anomalies occur across the year."""
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.fill_between(df[TIMESTAMP_COL], df[TARGET_COL],
                    alpha=0.7, color='#DD8452', step='post')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_title('Anomaly Occurrences Over Time', fontsize=13, fontweight='bold')
    ax.set_xlabel('Month (2018)')
    fig.tight_layout()
    _save(fig, 'eda', '05_anomaly_timeline.png')


def plot_normal_vs_anomaly_detail(df: pd.DataFrame):
    """Zoom into one anomaly event vs a normal period."""
    # Find first anomaly event
    anom_start_idx = df[df[TARGET_COL] == 1].index[0]
    # Normal window: 48 steps before anomaly
    norm_start = max(0, anom_start_idx - 48)
    norm_end   = anom_start_idx
    # Anomaly window: first 48 steps of anomaly
    anom_end   = min(len(df), anom_start_idx + 48)

    df_norm = df.iloc[norm_start:norm_end]
    df_anom = df.iloc[anom_start_idx:anom_end]

    sensors_to_show = PRESSURE_SENSORS[:2] + FLOW_SENSORS[:1]
    fig, axes = plt.subplots(len(sensors_to_show), 2, figsize=(14, 8))
    for i, col in enumerate(sensors_to_show):
        axes[i, 0].plot(df_norm[TIMESTAMP_COL], df_norm[col], color='#4C72B0', lw=1.2)
        axes[i, 0].set_title(f'{col} — Normal', fontsize=9)
        axes[i, 1].plot(df_anom[TIMESTAMP_COL], df_anom[col], color='#DD8452', lw=1.2)
        axes[i, 1].set_title(f'{col} — Anomaly', fontsize=9)
        for ax in axes[i]:
            ax.tick_params(labelsize=7)
    fig.suptitle('Normal vs Anomaly Period Detail', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '06_normal_vs_anomaly_detail.png')


def plot_rolling_stats(df: pd.DataFrame, sensor='n1', window=48):
    """Rolling mean and std for a representative sensor."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    roll_mean = df[sensor].rolling(window).mean()
    roll_std  = df[sensor].rolling(window).std()
    axes[0].plot(df[TIMESTAMP_COL], df[sensor],   lw=0.3, alpha=0.5, label='Raw')
    axes[0].plot(df[TIMESTAMP_COL], roll_mean,    lw=1.0, color='red', label=f'Rolling mean ({window})')
    axes[0].set_ylabel(f'{sensor} (m)')
    axes[0].legend(fontsize=8)
    axes[1].plot(df[TIMESTAMP_COL], roll_std, lw=0.8, color='orange')
    axes[1].set_ylabel(f'{sensor} Rolling Std')
    # Shade anomalies
    for ax in axes:
        anomaly_mask = df[TARGET_COL] == 1
        ax.fill_between(df[TIMESTAMP_COL], ax.get_ylim()[0], ax.get_ylim()[1],
                        where=anomaly_mask, alpha=0.1, color='red')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    fig.suptitle(f'Rolling Statistics for {sensor}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '07_rolling_statistics.png')


def plot_split_distributions(train_df, val_df, test_df):
    """Compare sensor distributions across train/val/test splits."""
    sensors_to_show = ['n1', 'p227', 'T1']
    fig, axes = plt.subplots(1, len(sensors_to_show), figsize=(14, 4))
    colors = {'Train': '#4C72B0', 'Val': '#55A868', 'Test': '#DD8452'}
    for ax, col in zip(axes, sensors_to_show):
        for name, split, color in [('Train', train_df, colors['Train']),
                                    ('Val',   val_df,   colors['Val']),
                                    ('Test',  test_df,  colors['Test'])]:
            ax.hist(split[col], bins=50, alpha=0.5, label=name,
                    color=color, density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle('Sensor Distributions Across Splits', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'eda', '08_split_distributions.png')


# ── Training plots ────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, train_metrics=None,
                         val_metrics=None, metric_name='F1',
                         model_name='model', subdir='training'):
    fig, axes = plt.subplots(1, 2 if train_metrics else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train', color='#4C72B0')
    axes[0].plot(epochs, val_losses,   label='Val',   color='#DD8452')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].legend()
    if train_metrics and len(axes) > 1:
        axes[1].plot(epochs, train_metrics, label=f'Train {metric_name}', color='#4C72B0')
        axes[1].plot(epochs, val_metrics,   label=f'Val {metric_name}',   color='#DD8452')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'{model_name} — {metric_name}')
        axes[1].legend()
    fig.tight_layout()
    _save(fig, subdir, f'{model_name}_training_curves.png')


def plot_confusion_matrix(cm, model_name='model', subdir='results'):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} — Confusion Matrix')
    fig.tight_layout()
    _save(fig, subdir, f'{model_name}_confusion_matrix.png')


def plot_roc_curve(fpr, tpr, auc_score, model_name='model', subdir='results'):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} — ROC Curve')
    ax.legend()
    fig.tight_layout()
    _save(fig, subdir, f'{model_name}_roc_curve.png')
