from evaluation import EvalMethod, CumulativePlotter, ResultAccumulator
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sdso_eval_vkitti = EvalMethod('/tmp', 'vkitti', 'SDSO')
    #sdso_eval_vkitti.evaluate_all()
    #sdso_eval_vkitti.save_results()
    sdso_eval_vkitti.load_results()

    sdso_eval_kitti = EvalMethod('/tmp', 'kitti', 'SDSO')
    #sdso_eval_kitti.evaluate_all()
    #sdso_eval_kitti.save_results()
    sdso_eval_kitti.load_results()

    sdso_all = ResultAccumulator()
    sdso_all.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, [25, 25, 25, 25])
    sdso_all.add_results('kitti', 'SDSO', sdso_eval_kitti.results_dict, 0.8, [25, 25, 25, 25])

    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot([data.ape_rmse for data in sdso_eval_vkitti.results_dict.values()])
    # axs[0, 1].plot([data.ape_rmse_rot for data in sdso_eval_vkitti.results_dict.values()])
    # axs[1, 0].plot([data.rpe_rmse for data in sdso_eval_vkitti.results_dict.values()])
    # axs[1, 1].plot([data.rpe_rmse_rot for data in sdso_eval_vkitti.results_dict.values()])
    # plt.show()
    #
    # key_list = [*sdso_eval_vkitti.results_dict.keys()]

    error_list_ape_t = np.arange(0, 15, 0.01)
    error_list_ape_r = np.arange(0, 5, 0.01)
    error_list_rpe_t = np.arange(0, 5, 0.01)
    error_list_rpe_r = np.arange(0, 1, 0.01)
    error_list = [error_list_ape_t, error_list_ape_r, error_list_rpe_t, error_list_rpe_r]

    plotter = CumulativePlotter()
    plotter.add_data('SDSO', sdso_all, 'r', error_list)
    plotter.plot_figure()

