from evaluation import Results, EvalMethod, CumulativePlotter
import numpy as np

if __name__ == "__main__":
    sdso_eval_vkitti = EvalMethod('/tmp', 'vkitti', 'SDSO')
    #sdso_eval_vkitti.evaluate_all()
    #sdso_eval_vkitti.save_results()

    sdso_eval_vkitti.load_results()
    #sdso_eval_vkitti.viz_matrix_ape_rmse(0.8, 25)

    viz_plot = CumulativePlotter()
    viz_plot.add_results('vkitti', 'SDSO', sdso_eval_vkitti.results_dict, 0.8, 25)
    error_list = np.arange(0, 200, 0.01)
    viz_plot.plot_cumulative_error(error_list)
