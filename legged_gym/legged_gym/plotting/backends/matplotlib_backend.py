import os
from datetime import datetime


class MatplotlibBackend:
    """Render recorded scalar series with matplotlib."""

    def render(self, times, series, figures=None, output_dir=None, show=True):
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures = figures or self._default_figures(series)
        saved_paths = []
        for figure_cfg in figures:
            fig = self._render_figure(plt, times, series, figure_cfg)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                name = figure_cfg.get("name", "plot")
                path = os.path.join(output_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                fig.savefig(path, dpi=160, bbox_inches="tight")
                saved_paths.append(path)
            if not show:
                plt.close(fig)

        if show:
            plt.show()
        return saved_paths

    def _render_figure(self, plt, times, series, figure_cfg):
        plot_cfgs = figure_cfg.get("plots", [])
        if not plot_cfgs:
            plot_cfgs = [{"title": key, "series": [key]} for key in sorted(series.keys())]

        rows, cols = self._layout(figure_cfg, len(plot_cfgs))
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(5.0 * cols, 3.2 * rows))
        fig.suptitle(figure_cfg.get("title", figure_cfg.get("name", "plot")))

        for index, plot_cfg in enumerate(plot_cfgs):
            ax = axes[index // cols][index % cols]
            for key in plot_cfg.get("series", []):
                if key not in series:
                    continue
                ax.plot(times[: len(series[key])], series[key], label=key)
            ax.set_title(plot_cfg.get("title", ""))
            ax.set_xlabel("time [s]")
            ax.grid(True, alpha=0.3)
            if plot_cfg.get("ylabel"):
                ax.set_ylabel(plot_cfg["ylabel"])
            if plot_cfg.get("series"):
                ax.legend(loc="best")

        for index in range(len(plot_cfgs), rows * cols):
            axes[index // cols][index % cols].axis("off")
        fig.tight_layout()
        return fig

    def _layout(self, figure_cfg, num_plots):
        layout = figure_cfg.get("layout", None)
        if layout is not None:
            return int(layout[0]), int(layout[1])
        cols = 1 if num_plots <= 3 else 2
        rows = (num_plots + cols - 1) // cols
        return max(rows, 1), max(cols, 1)

    def _default_figures(self, series):
        return [
            {
                "name": "plot",
                "plots": [{"title": key, "series": [key]} for key in sorted(series.keys())],
            }
        ]
