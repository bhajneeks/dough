'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        print('evaluating macro precision...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        score = precision_score(
            true_y,
            pred_y,
            average='macro',
            zero_division=0
        )

        return score