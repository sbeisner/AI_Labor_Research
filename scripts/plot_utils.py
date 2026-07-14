"""Shared display utilities for manuscript plot scripts.

When running inside Jupyter/Quarto: figures are displayed inline.
When running standalone (e.g. `python scripts/foo_plot.py`): figures are
saved as PNGs to output/preview/ for quick visual iteration.
"""
import pathlib


def in_jupyter() -> bool:
    """Return True if running inside a live IPython/Jupyter kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False


def setup_matplotlib() -> None:
    """Set the correct matplotlib backend before pyplot is imported.
    Must be called before `import matplotlib.pyplot`.
    """
    import matplotlib
    if in_jupyter():
        matplotlib.use('module://matplotlib_inline.backend_inline')
    else:
        matplotlib.use('Agg')


def show_or_save(fig, name: str) -> None:
    """Display inline in Jupyter/Quarto, or save PNG to output/preview/ standalone."""
    if in_jupyter():
        from IPython.display import display
        display(fig)
    else:
        out = pathlib.Path('output/preview')
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'{name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[preview] Saved {path}")
