# %%

import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results


# %%

log_dir = "../../../../pvcvolume/"
plot_results([log_dir], 1_000_000, results_plotter.X_TIMESTEPS, "SAC Protagonist AV")
plt.show()

# %%
