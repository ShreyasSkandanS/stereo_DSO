from evaluation import EvalMethod, CumulativePlotter, ResultAccumulator
import numpy as np
import matplotlib.pyplot as plt


def load_evaluation(basedir: str, dataset: str, method: str, normalize_traj: bool) -> EvalMethod:
    evaluator = EvalMethod(basedir, dataset, method)
    if (evaluator.save_dir / "evaluate.pkl").is_file():
        print(f"Loading {method} - {dataset} data from pickle.")
        evaluator.load_results()
    else:
        print(f"Generating {method} - {dataset} data and saving to pickle.")
        evaluator.evaluate_all(normalize_trajectory=normalize_traj)
        evaluator.save_results()
    return evaluator


if __name__ == "__main__":
    traj_norm = True

    # LOAD DATA

    sdso_eval_vkitti = load_evaluation('/tmp', 'vkitti', 'SDSO', normalize_traj=traj_norm)
    # sdso_eval_vkitti.viz_rmse()
    sdso_eval_kitti = load_evaluation('/tmp', 'kitti', 'SDSO', normalize_traj=traj_norm)
    # sdso_eval_kitti.viz_rmse()
    sdso_eval_tta = load_evaluation('/tmp', 'tartan_air', 'SDSO', normalize_traj=traj_norm)
    # sdso_eval_tta.viz_rmse()

    dsol_eval_vkitti = load_evaluation('/tmp', 'vkitti', 'DSOL', normalize_traj=traj_norm)
    # dsol_eval_vkitti.viz_rmse()
    dsol_eval_kitti = load_evaluation('/tmp', 'kitti', 'DSOL', normalize_traj=traj_norm)
    # dsol_eval_kitti.viz_rmse()
    dsol_eval_tta = load_evaluation('/tmp', 'tartan_air', 'DSOL', normalize_traj=traj_norm)
    # dsol_eval_tta.viz_rmse()

    # ACCUMULATE DATA

    sdso_all = ResultAccumulator()
    sdso_all.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    sdso_all.add_results('kitti', 'SDSO', sdso_eval_kitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    sdso_all.add_results('tartan_air', 'SDSO', sdso_eval_tta.results_dict, 0.8, [0.50, 10, 0.50, 10])

    dsol_all = ResultAccumulator()
    dsol_all.add_results('vkitti', 'DSOL', dsol_eval_vkitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    dsol_all.add_results('kitti', 'DSOL', dsol_eval_kitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    dsol_all.add_results('tartan_air', 'DSOL', dsol_eval_tta.results_dict, 0.8, [0.50, 10, 0.50, 10])

    # #key_list = [*sdso_eval_vkitti.results_dict.keys()]

    # PLOT CUMULATIVE DATA

    error_list_ape_t = np.arange(0, 0.1, 0.00001)
    error_list_ape_r = np.arange(0, 90, 0.01)
    error_list_rpe_t = np.arange(0, 0.1, 0.00001)
    error_list_rpe_r = np.arange(0, 5, 0.01)
    error_list = [error_list_ape_t, error_list_ape_r, error_list_rpe_t, error_list_rpe_r]

    plotter = CumulativePlotter(traj_norm=traj_norm)
    plotter.add_data('SDSO', sdso_all, 'r', error_list)
    # plotter.add_data('DSOL', dsol_all, 'g', error_list)
    plotter.add_data('DSOL', dsol_all, 'b', error_list)

    for ax in plotter.axs.ravel():
        ax.set_ylim([0, 101])
    ylimb = 0
    plotter.axs[0, 0].set_xlim([ylimb, 10])
    plotter.axs[0, 1].set_xlim([ylimb, 60])
    plotter.axs[1, 0].set_xlim([ylimb, 1])
    plotter.axs[1, 1].set_xlim([ylimb, 3])

    plotter.plot_figure(save_fig=True)
