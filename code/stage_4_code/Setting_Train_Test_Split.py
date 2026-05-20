'''
Setting for Stage 4 — wires together dataset, method, result, evaluate.
Same pattern as Stage 3, works for both classification and generation.
'''

from code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    def load_run_save_evaluate(self):
        # load the data (classification or generation, depending on the loader)
        loaded_data = self.dataset.load()

        # hand data to the method and run it
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save predictions
        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()

        # evaluate (only makes sense for classification — generation skips this)
        if self.evaluate is not None and 'pred_y' in learned_result:
            self.evaluate.data = learned_result
            return self.evaluate.evaluate(), learned_result
        else:
            return None, learned_result
