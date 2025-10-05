import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def remove_unused_axes(fig, axes, used):
    """Remove eixos não usados (vazios) do figure."""
    for j in range(used, len(axes)):
        fig.delaxes(axes[j])

def plot_distributions(df, quantitative_cols, ordinal_cols, nominal_cols,
                       title_suffix="", force_ordinal_continuous=False,
                       delay_log_scale=True):
    """
    Plota distribuições. 
    *********************ATENÇÃO:*********************
    Para colunas de 'delay' aplicamos np.log1p(clip(lower=0))
    para PRESERVAR zeros e comprimir caudas (equivalente a log(1+x)). Log(0) não existe, não podemos ignorar a maioria dos dados.
    """
    sns.set_style("white")

    if quantitative_cols:
        n = len(quantitative_cols)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, max(1, n_rows)*4))
        axes = axes.flatten()
        for i, col in enumerate(quantitative_cols):
            ax = axes[i]
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if delay_log_scale and "delay" in col.lower():
                series_plot = np.log1p(series.clip(lower=0))
                sns.histplot(series_plot, kde=False, ax=ax, stat="percent", bins=40)
                ax.set_xlabel(f"{col} (log1p)")
            else:
                sns.histplot(series, kde=False, ax=ax, stat="percent", bins=40)
            ax.set_title(col)
            ax.set_ylabel("Percent")
            ax.tick_params(axis="x", rotation=45)
        remove_unused_axes(fig, axes, len(quantitative_cols))
        fig.suptitle(f"Distribuição das Variáveis Quantitativas {title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(); plt.show()

    # --- Ordinais ---
    if ordinal_cols:
        n = len(ordinal_cols)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, max(1, n_rows)*4))
        axes = axes.flatten()
        for i, col in enumerate(ordinal_cols):
            ax = axes[i]
            coerced = pd.to_numeric(df[col], errors="coerce").dropna()
            if force_ordinal_continuous:
                sns.histplot(coerced, kde=False, ax=ax, stat="percent", bins=40)
            else:
                counts = coerced.value_counts(normalize=True).sort_index() * 100
                sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
            ax.set_title(col)
            ax.set_ylabel("Percent")
            ax.tick_params(axis="x", rotation=45)
        remove_unused_axes(fig, axes, len(ordinal_cols))
        fig.suptitle(f"Distribuição das Variáveis Ordinais {title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(); plt.show()

    # --- Nominais ---
    if nominal_cols:
        # se lista vazia ou None, pula todo o bloco (evita n_rows=0)
        n = len(nominal_cols)
        if n > 0:
            n_cols = 3
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, max(1, n_rows)*4))
            axes = axes.flatten()
            used = 0
            for i, col in enumerate(nominal_cols):
                ax = axes[i]
                counts = df[col].astype(str).value_counts(dropna=False)
                labels = list(counts.index)
                sns.barplot(x=labels, y=(counts/counts.sum()*100).values, ax=ax)
                ax.set_title(col)
                ax.set_ylabel("Percent")
                ax.tick_params(axis="x", rotation=45)
                used += 1
            remove_unused_axes(fig, axes, used)
            fig.suptitle(f"Distribuição das Variáveis Nominais {title_suffix}", fontsize=16, y=1.02)
            plt.tight_layout(); plt.show()
