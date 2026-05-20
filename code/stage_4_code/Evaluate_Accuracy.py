'''
Accuracy metric — same as Stage 3.
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating accuracy...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])
