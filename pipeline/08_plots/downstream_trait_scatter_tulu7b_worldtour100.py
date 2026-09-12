"""Tulu-2-7B configuration for the fixed 100-user scatter generator."""
import os

import downstream_trait_scatter_vicuna13b_worldtour100 as plots


plots.SHORT = "tulu-2-7b"
plots.OUT = os.path.join(
    plots.HERE, "g_plots", "Tulu-2", plots.SHORT, "downstream", "100_users")
plots.OCEAN_ROOT = os.path.join(plots.HERE, "results_worldtour_tulu7b_100")
plots.DS_ROOT = os.path.join(
    plots.HERE, "results_downstream_worldtour_tulu7b_100")
plots.BASELINE_OCEAN = os.path.join(plots.HERE, "results_tulu-2-7b_baseline.json")

if __name__ == "__main__":
    plots.main()
