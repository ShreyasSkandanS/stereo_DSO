from evaluation import Results, EvalMethod, CumulativePlotter, ResultAccumulator
import numpy as np

if __name__ == "__main__":
    sdso_eval_vkitti = EvalMethod('/tmp', 'vkitti', 'SDSO')
    # sdso_eval_vkitti.evaluate_all()
    # sdso_eval_vkitti.save_results()

    sdso_eval_vkitti.load_results()
    # sdso_eval_vkitti.viz_matrix_ape_rmse(0.8, 25)

    sdso_all = ResultAccumulator()
    sdso_all.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, [25, 25, 25, 25])
    sdso_all.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, [25, 25, 25, 25])
    sdso_all.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, [25, 25, 25, 25])

    error_list = np.arange(0, 2, 0.01)

    plotter = CumulativePlotter()
    plotter.add_data(sdso_all, 'r', error_list)
    plotter.plot_figure()

