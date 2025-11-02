import os, matplotlib.pyplot as plt

def bar_plot(df, xcol, ycol, title, outfile, results_dir, y_min=None, y_max=None):
    ax = df.plot(kind="bar", x=xcol, y=ycol, legend=False)
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel(ycol)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)  # ← fixe les bornes Y
    fig = ax.get_figure(); fig.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, outfile)
    fig.savefig(path, dpi=150); plt.close(fig)
    print("Saved:", path)