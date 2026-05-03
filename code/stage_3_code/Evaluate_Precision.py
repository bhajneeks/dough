'''
Concrete Evaluate class for macro precision.
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        print('evaluating macro precision...')
        return precision_score(
            self.data['true_y'],
            self.data['pred_y'],
            average='macro',
            zero_division=0,
        )
