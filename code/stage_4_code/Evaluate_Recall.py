'''
Recall metric — binary since IMDB is pos/neg.
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import recall_score


class Evaluate_Recall(evaluate):
    data = None

    def evaluate(self):
        print('evaluating recall...')
        return recall_score(
            self.data['true_y'],
            self.data['pred_y'],
            average='binary',
            zero_division=0,
        )
