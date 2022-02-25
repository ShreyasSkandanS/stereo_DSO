from evaluation import Results, EvalMethod

if __name__ == "__main__":
    sdso_eval_vkitti = EvalMethod('/tmp', 'vkitti', 'SDSO')
    #sdso_eval_vkitti.evaluate_all()
    #sdso_eval_vkitti.save_results()

    sdso_eval_vkitti.load_results()
    sdso_eval_vkitti.visualize_results(0.8, 25)
