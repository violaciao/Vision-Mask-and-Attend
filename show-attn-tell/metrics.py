import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score


class ConfusionMatrix(object):

    def __init__(self, num_classes):
        '''
        num_classes: int, number of classes
        '''
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros([num_classes, num_classes]).astype(int)


    def add(self, y_pred, y_target):
        '''
        Update confusion matrix given target value and predicted value

        y_pred: int, list, or numpy array, predicted value
        y_target: int, list, or numpy array, target value
        '''
        if isinstance(y_pred, Variable):
            y_pred = y_pred.data
        if isinstance(y_target, Variable):
            y_target = y_target.data

        if not hasattr(y_pred, "__len__") and not hasattr(y_target, "__len__"):
            # add single value pairs
            self.confusion_matrix[y_pred, y_target] += 1

        assert len(y_pred) == len(y_target), \
                'lengths of two arrays must be equal,' + \
                'length %d does not match length %d' % (len(y_pred), len(y_target))

        for i in range(len(y_pred)):
            # add batch
            self.confusion_matrix[int(y_pred[i]), int(y_target[i])] += 1


    @property
    def acc(self):
        return sum(self.confusion_matrix[i][i] 
                for i in range(self.num_classes)) / self.confusion_matrix.sum()


    def __str__(self):
        s = 'Confusion Matrix with %d classes\n' % self.num_classes
        s += '\t\t\tTruth\n'
        for i in range(self.num_classes):
            s += ' %s' % ('Predicted' if i == 0 else '\t')

            for j in range(self.num_classes):
                s += '\t%d' % self.confusion_matrix[i,j]
            s += '\n'

        return s[:-1]


class AUC(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y_true = None    # true labels
        self.y_score = None   # target scores


    def add(self, y_true, y_score):
        assert len(y_true) == len(y_score), \
                'lengths of two arrays must be equal,' + \
                'length %d does not match length %d' % (len(y_true), len(y_score))

        if type(y_true) == Variable:
            y_true = y_true.data

        if type(y_score) == Variable:
            y_score = y_score.data

        label = np.zeros([len(y_true), self.num_classes])
        for i in range(len(y_true)):
            label[i, y_true[i]] = 1

        if self.y_true is None:
            self.y_true = label
            self.y_score = y_score
        else:
            self.y_true = np.concatenate((self.y_true, label))
            self.y_score = np.concatenate((self.y_score, y_score))


    def __str__(self):
        return 'AUC score: {:.4f}'.format(
                roc_auc_score(self.y_true, self.y_score))


class Majority(object):

    def __init__(self):
        self.votes = None


    def add(self, candidates):
        if isinstance(candidates, Variable):
            candidates = candidates.data

        candidates = candidates.cpu().numpy()

        if self.votes is None:
            self.votes = candidates
        self.votes = np.concatenate((self.votes, candidates), axis=1)


    def vote_result(self):
        result = []
        for i in range(len(self.votes)):
            counts = np.bincount(self.votes[i])
            major = np.argmax(counts)
            result.append(major)
        return result

