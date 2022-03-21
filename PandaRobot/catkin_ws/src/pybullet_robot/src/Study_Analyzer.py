import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


study_name = "ppo-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)

'''
plot_optimization_history(loaded_study).show(renderer="browser")
plot_parallel_coordinate(loaded_study).show(renderer="browser")
plot_param_importances(loaded_study).show(renderer="browser")
'''

fig1 = plot_optimization_history(loaded_study)
fig2 = plot_parallel_coordinate(loaded_study)
fig3 = plot_param_importances(loaded_study)

fig1.show()
fig2.show()
fig3.show()