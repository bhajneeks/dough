'''
Precision metric — binary since IMDB is pos/neg.
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        print('evaluating precision...')
        return precision_score(
            self.data['true_y'],
            self.data['pred_y'],
            average='binary',
            zero_division=0,
        )
