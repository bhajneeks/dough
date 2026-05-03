'''
Concrete Evaluate class for macro F1.
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        print('evaluating macro f1...')
        return f1_score(
            self.data['true_y'],
            self.data['pred_y'],
            average='macro',
            zero_division=0,
        )
