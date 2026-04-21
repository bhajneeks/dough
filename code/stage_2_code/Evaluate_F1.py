'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        print('evaluating macro f1...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        score = f1_score(
            true_y,
            pred_y,
            average='macro',
            zero_division=0
        )

        return score