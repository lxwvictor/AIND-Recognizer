import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Started my_model_selectors.py")

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        try:
            # Initialize BIC to positive inf, as we need to find the lowest BIC.
            bestBIC = curBIC = float("inf")
            for n_components in range(self.min_n_components, self.max_n_components+1):
                try:
                    hmm_model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = hmm_model.score(self.X, self.lengths)
                    #print("word {}, n {}, logL {}".format(self.this_word, n_components, logL))

                    # N is the number of data points
                    N = len(self.X)
                    # p is the number of parameters, let m=n_components, f=num_features
                    # size of transmat matrix less one row, m*(m-1), plus
                    # free starting probabilities, m-1, plus
                    # number of means, m*f, plus
                    # number of covariances, m*f
                    # equals, m^2 + 2*m*f - 1
                    p = n_components ** 2 + 2*n_components*hmm_model.n_features - 1
                    curBIC = -2 * logL + p * math.log(N)
                    if curBIC < bestBIC:
                        bestBIC = curBIC
                        bestModel = hmm_model
                except:
                    continue
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return bestModel
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        try:
            # Initialize DIC to nagative inf, as we need to find the biggest DIC.
            bestDIC = curDIC = -float("inf")
            M = len(self.hwords)
            sumOthers = 0
            for n_components in range(self.min_n_components, self.max_n_components+1):
                try:
                    hmm_model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = hmm_model.score(self.X, self.lengths)
                    #print("word {}, n {}, logL {}".format(self.this_word, n_components, logL))

                    for word in self.hwords:
                        if self.this_word == word:
                            continue
                        X, lengths = self.hwords[word]
                        otherLogL = hmm_model.score(X, lengths)
                        sumOthers += otherLogL

                    #print("sum other work {}".format(sumOthers))
                    curDIC = logL - sumOthers/(M-1)
                    if curDIC > bestDIC:
                        bestDIC = curDIC
                        bestModel = hmm_model
                except:
                    continue
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return bestModel
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        from sklearn.model_selection import KFold

        try:
            # Initialize CV to negative inf, as we need to find the biggest CV.
            bestCV = curCV = -float("inf")
            for n_components in range(self.min_n_components, self.max_n_components+1):
                foldCount = foldLogSum = 0

                # n_splits of KFold is default 3, at least 2
                if len(self.sequences) == 2:
                    split_method = KFold(n_splits=2)
                elif len(self.sequences) < 2:
                    continue
                else:
                    split_method = KFold()
                for cv_train_index, cv_test_index in split_method.split(self.sequences):
                    trainX, train_lengths = combine_sequences(cv_train_index, self.sequences)
                    testX, test_lengths = combine_sequences(cv_test_index, self.sequences)
                    try:
                        hmm_model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(trainX, train_lengths)
                        testLogL = hmm_model.score(testX, test_lengths)
                        foldCount += 1
                        foldLogSum += testLogL
                        #print("word {}, n {}, logL {}".format(self.this_word, n_components, testLogL))
                    except:
                        continue
                if foldCount != 0:  # some data cannot fit into the model
                    CVLogL = foldLogSum/foldCount
                    curCV = CVLogL
                    #rint("average CV {}".format(curCV))
                    if curCV > bestCV:
                        bestCV = curCV
                        bestModel = hmm_model
                else:
                    continue

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return bestModel
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
        raise NotImplementedError

logging.info("Finished my_model_selectors.py")