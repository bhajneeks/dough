'''
F1 metric — binary since IMDB is pos/neg.
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        print('evaluating f1...')
        return f1_score(
            self.data['true_y'],
            self.data['pred_y'],
            average='binary',
            zero_division=0,
        )
