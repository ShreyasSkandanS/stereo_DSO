from evaluation import EvalMethod, CumulativePlotter, ResultAccumulator
import numpy as np
import matplotlib.pyplot as plt
from eval_methods import load_evaluation

if __name__ == "__main__":
    traj_norm = True

    # LOAD DATA
    sdso_eval_vkitti = load_evaluation('/tmp', 'vkitti', 'SDSO', normalize_traj=traj_norm)
    sdso_eval_kitti = load_evaluation('/tmp', 'kitti', 'SDSO', normalize_traj=traj_norm)
    sdso_eval_tta = load_evaluation('/tmp', 'tartan_air', 'SDSO', normalize_traj=traj_norm)

    dsol_eval_vkitti = load_evaluation('/tmp', 'vkitti', 'DSOL', normalize_traj=traj_norm)
    dsol_eval_kitti = load_evaluation('/tmp', 'kitti', 'DSOL', normalize_traj=traj_norm)
    dsol_eval_tta = load_evaluation('/tmp', 'tartan_air', 'DSOL', normalize_traj=traj_norm)

    # ACCUMULATE DATA
    sdso_all = ResultAccumulator()
    sdso_all.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    sdso_all.add_results('kitti', 'SDSO', sdso_eval_kitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    sdso_all.add_results('tartan_air', 'SDSO', sdso_eval_tta.results_dict, 0.8, [0.50, 10, 0.50, 10])

    dsol_all = ResultAccumulator()
    dsol_all.add_results('vkitti', 'DSOL', dsol_eval_vkitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    dsol_all.add_results('kitti', 'DSOL', dsol_eval_kitti.results_dict, 0.8, [0.50, 10, 0.50, 10])
    dsol_all.add_results('tartan_air', 'DSOL', dsol_eval_tta.results_dict, 0.8, [0.50, 10, 0.50, 10])

    n_rows = 16
    n_cols = 17
    ape_t_amax = 0.1
    ape_r_amax = 10

    sdso_ape_t_arr = np.zeros(n_rows * n_cols, dtype=float) - 1
    sdso_ape_t_arr[:len(sdso_all.ape_rmse_tr)] = np.asarray(sdso_all.ape_rmse_tr)
    sdso_ape_t = sdso_ape_t_arr.reshape([n_rows, n_cols])
    sdso_ape_t = np.clip(sdso_ape_t, a_min=None, a_max=ape_t_amax)
    sdso_ape_t = np.ma.masked_where(sdso_ape_t == -1, sdso_ape_t)

    sdso_ape_r_arr = np.zeros(n_rows * n_cols, dtype=float) - 1
    sdso_ape_r_arr[:len(sdso_all.ape_rmse_rot)] = np.asarray(sdso_all.ape_rmse_rot)
    sdso_ape_r = sdso_ape_r_arr.reshape([n_rows, n_cols])
    sdso_ape_r = np.clip(sdso_ape_r, a_min=None, a_max=ape_r_amax)
    sdso_ape_r = np.ma.masked_where(sdso_ape_r == -1, sdso_ape_r)

    dsol_ape_t_arr = np.zeros(n_rows * n_cols, dtype=float) - 1
    dsol_ape_t_arr[:len(dsol_all.ape_rmse_tr)] = np.asarray(dsol_all.ape_rmse_tr)
    dsol_ape_t = dsol_ape_t_arr.reshape([n_rows, n_cols])
    dsol_ape_t = np.clip(dsol_ape_t, a_min=None, a_max=ape_t_amax)
    dsol_ape_t = np.ma.masked_where(dsol_ape_t == -1, dsol_ape_t)

    dsol_ape_r_arr = np.zeros(n_rows * n_cols, dtype=float) - 1
    dsol_ape_r_arr[:len(dsol_all.ape_rmse_rot)] = np.asarray(dsol_all.ape_rmse_rot)
    dsol_ape_r = dsol_ape_r_arr.reshape([n_rows, n_cols])
    dsol_ape_r = np.clip(dsol_ape_r, a_min=None, a_max=ape_r_amax)
    dsol_ape_r = np.ma.masked_where(dsol_ape_r == -1, dsol_ape_r)

    # fig2, ax2 = plt.subplots()
    # ax2.hist(np.clip(np.asarray(sdso_all.ape_rmse_tr), 0, 1))
    # plt.show()

    fig, axs = plt.subplots(2, 2)

    colormap = 'turbo'

    im00 = axs[0, 0].imshow(sdso_ape_t * 100, cmap=colormap, interpolation='nearest')
    axs[0, 0].grid(False)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticklabels([])
    axs[0, 0].set_ylabel('SDSO')

    im01 = axs[0, 1].imshow(sdso_ape_r, cmap=colormap, interpolation='nearest')
    axs[0, 1].grid(False)
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_yticklabels([])

    im10 = axs[1, 0].imshow(dsol_ape_t * 100, cmap=colormap, interpolation='nearest')
    axs[1, 0].grid(False)
    axs[1, 0].set_xticklabels([])
    axs[1, 0].set_yticklabels([])
    axs[1, 0].set_ylabel('DSOL')
    axs[1, 0].set_xlabel('Translational APE (%)')

    im11 = axs[1, 1].imshow(dsol_ape_r, cmap=colormap, interpolation='nearest')
    axs[1, 1].grid(False)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].set_xlabel('Rotational APE (deg)')

    # cax00 = fig.add_axes(
    #     [axs[0, 0].get_position().x1 + 0.01, axs[0, 0].get_position().y0, 0.02, axs[0, 0].get_position().height])
    # cax01 = fig.add_axes(
    #     [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.02, axs[0, 1].get_position().height])
    # cax10 = fig.add_axes(
    #     [axs[1, 0].get_position().x1 + 0.01, axs[1, 0].get_position().y0, 0.02, axs[1, 0].get_position().height])
    # cax11 = fig.add_axes(
    #     [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.02, axs[1, 1].get_position().height])
    #
    # fig.colorbar(im00, ax=axs[0, 0], cax=cax00)
    # fig.colorbar(im01, ax=axs[0, 1], cax=cax01)
    # fig.colorbar(im10, ax=axs[1, 0], cax=cax10)
    # fig.colorbar(im11, ax=axs[1, 1], cax=cax11)

    fig.colorbar(im00, ax=axs[0, 0])
    fig.colorbar(im01, ax=axs[0, 1])
    fig.colorbar(im10, ax=axs[1, 0])
    fig.colorbar(im11, ax=axs[1, 1])

    #plt.show()
    plt.tight_layout()
    plt.savefig("/tmp/error_matrix_all_turbo.pdf", bbox_inches='tight')
