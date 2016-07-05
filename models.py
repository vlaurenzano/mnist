from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from functools import reduce
import pickle

def load_classifier(algo, config):
    """ Load our classifier

        Initializes our Persistent Classifier with a classifier we explicitly support.
        Classifier parameters are taken from our config file.

        Args:
            algo (str): The algorithm we want to use
            config (dict): A dictionary type object that has the property get and our configs
        Raises:
            Exception
    """
    if algo == 'RandomForest':
        n_estimators = int(config.get('n_estimators', 20))
        n_jobs = int(config.get('n_jobs', 4))
        return PersistentClassifier(RandomForestClassifier,n_estimators=n_estimators, n_jobs=n_jobs)

    elif algo == 'KNN':
        n_neighbors = int(config.get('n_neighbors', 5))
        return PersistentClassifier(KNeighborsClassifier, n_neighbors=n_neighbors)

    elif algo == 'LogisticRegression':
        return PersistentClassifier(LogisticRegression)

    raise 'Improper algorithm value give, see --help'

class PersistentClassifier():
    """
    A class to persist and load our classifiers

    This class uses relative paths to store and load data from the bin folder.
    This class wraps sklearn classifiers and persists to a pickle file stored under a hash made from it's
    constructor arguments.
    This class is lazy loads the classifier so it can take it from memory when appropriate.

    Args:
        classifier_class (ABCMeta): A meta class that works with this class. Classifiers from sklearn

    Attributes:
        hashkey (str): A key to load and persist data to
        classifier_constructor (lambda): A constructor for creating the classifier
        classifier (object): Our constructed classifier

    """
    def __init__(self, classifier_class, **kwargs):
        name = classifier_class.__name__
        self.hashkey = name + reduce(lambda carry, key: carry + key + "-" + str(kwargs[key]), sorted(kwargs), "-")
        self.classifier_constructor = lambda : classifier_class(**kwargs)
        self.classifier = None

    def load(self):
        """ Loads our classifier from memory
        Returns:
             classifier (object): a classifier
        Raises:
            FileNotFoundException: if the file is not found
        """
        self.classifier = pickle.load(open( "bin/" + self.hashkey +".p", "rb" ))
        return self.classifier

    def fit(self,features, classes, persist=True):
        """ Fits our model and persists it
        Args:
            features(np.array): our feature set
            classes(np.array): our classes corresponding to features
        """
        self.classifier = self.classifier_constructor()
        self.classifier.fit(features,classes)
        pickle.dump(self.classifier, open("bin/" + self.hashkey + ".p", "wb" ))

    def predict(self, data):
        """ Fits our model and persists it
        Args:
            data(np.array): our feature set
        Returns:
            np.array: our result set
        """
        return self.load().predict(data)

    def score(self, features, classes):
        """ Returns our accuracy
        Args:
            features(np.array): our feature set
            classes(np.array): our classes corresponding to features
        """
        return self.classifier.score(features,classes)

