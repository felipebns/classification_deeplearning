import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distributions(df, quantitative_cols, ordinal_cols, nominal_cols,
                       title_suffix="", force_ordinal_continuous=False):
    """
    Plota distribuições:
     - quantitativas: histograma (percent)
     - ordinais: se force_ordinal_continuous=True -> histograma (útil pós-zscore)
                else -> se poucos valores únicos (<=10) -> barplot por valor (discreto)
                        caso contrário -> histograma
     - nominais: se coluna original existe -> barplot; se só existem dummies (prefix-) -> soma dummies
    """
    # sem grid
    sns.set_style("white")

    # --- função auxiliar para remover eixos extras ---
    def clean_axes(fig, axes, used):
        for j in range(used, len(axes)):
            fig.delaxes(axes[j])

    # --- Quantitativas ---
    if quantitative_cols:
        n = len(quantitative_cols)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten()
        for i, col in enumerate(quantitative_cols):
            ax = axes[i]
            if col not in df.columns: continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty: continue
            sns.histplot(series, kde=False, ax=ax, stat="percent", bins=40)
            ax.set_title(col)
            ax.set_ylabel("Percent")
        clean_axes(fig, axes, len(quantitative_cols))
        fig.suptitle(f"Distribuição das Variáveis Quantitativas {title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(); plt.show()

    # --- Ordinais ---
    if ordinal_cols:
        n = len(ordinal_cols)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten()
        for i, col in enumerate(ordinal_cols):
            ax = axes[i]
            if col not in df.columns: continue
            coerced = pd.to_numeric(df[col], errors="coerce").dropna()

            if force_ordinal_continuous:
                if coerced.empty: continue
                sns.histplot(coerced, kde=False, ax=ax, stat="percent", bins=40)
            else:
                nunique = coerced.nunique()
                if coerced.empty:
                    counts = df[col].astype(str).value_counts()
                    sns.barplot(x=counts.index, y=(counts/counts.sum()*100).values, ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                elif nunique <= 10:
                    vals = coerced.round(0).astype(int)
                    counts = vals.value_counts().reindex(sorted(vals.unique()), fill_value=0)
                    sns.barplot(x=[str(x) for x in counts.index], y=(counts/counts.sum()*100).values, ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                else:
                    sns.histplot(coerced, kde=False, ax=ax, stat="percent", bins=40)

            ax.set_title(col)
            ax.set_ylabel("Percent")
        clean_axes(fig, axes, len(ordinal_cols))
        fig.suptitle(f"Distribuição das Variáveis Ordinais {title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(); plt.show()

        # --- Nominais ---
    if nominal_cols:
        # checar se existe pelo menos uma nominal (original ou dummy correspondente) no DF
        any_nominal = False
        for col in nominal_cols:
            if col in df.columns:
                any_nominal = True
                break
            prefix = col + "-"
            if any(c.startswith(prefix) for c in df.columns):
                any_nominal = True
                break

        if any_nominal:  # só plota se achou algo
            n = len(nominal_cols)
            n_cols = 3
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
            axes = axes.flatten()
            used = 0
            for i, col in enumerate(nominal_cols):
                ax = axes[i]
                if col in df.columns:
                    counts = df[col].astype(str).value_counts(dropna=False)
                    sns.barplot(x=counts.index, y=(counts/counts.sum()*100).values, ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    used += 1
                else:
                    prefix = col + "-"
                    dummy_cols = [c for c in df.columns if c.startswith(prefix)]
                    if not dummy_cols: 
                        ax.set_visible(False)
                        continue
                    counts_series = df[dummy_cols].sum(axis=0)
                    labels = [c[len(prefix):] for c in dummy_cols]
                    counts_pct = (counts_series / counts_series.sum() * 100).values
                    sns.barplot(x=labels, y=counts_pct, ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    used += 1
                ax.set_title(col)
                ax.set_ylabel("Percent")
            # remover eixos sobrando
            for j in range(used, len(axes)):
                fig.delaxes(axes[j])
            fig.suptitle(f"Distribuição das Variáveis Nominais {title_suffix}", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()

