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

    for sequence, length in test_set.get_all_Xlengths().values():

        best_score = float("-inf")
        best_word = None

        word_probabilities = {}

        for word, model in models.items():
            try:
                word_probabilities[word] = model.score(sequence, length)
            
                if word_probabilities[word] > best_score:
                
                    best_score = word_probabilities[word]
                    best_word = word
            except:
                word_probabilities[word] = float("-inf")

        guesses.append(best_word)
        probabilities.append(word_probabilities)

    return probabilities, guesses