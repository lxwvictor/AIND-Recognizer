import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # get_all_Xlengths() will return a dictionary, key is the index from 0,
    # value is a list [X, Xlengths]
    test_sequences = test_set.get_all_Xlengths()
    for test_index in range(len(test_sequences)):
        # Do not here the type of test_sequences is dict, test_index is an
        # integer just happens to be the key value
        test_X, test_Xlengths = test_sequences[test_index]
        test_word = test_set.wordlist[test_index]
        score_dict = {}
        for word, model in models.items():
            try:
                logL = model.score(test_X, test_Xlengths)
                score_dict[word] = logL
            except:
                # set the core to negative infinity if the model can't fit
                score_dict[word] = -float("inf")
                continue

        probabilities.append(score_dict)
        guessWord = max(score_dict, key=score_dict.get)
        guesses.append(guessWord)

    return probabilities, guesses
    raise NotImplementedError
