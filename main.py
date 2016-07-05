import numpy as np
import pandas as pd
from sklearn import cross_validation
import argparse
import configparser
import models


def fit(classifier, data):
    """ Fits the model
        Args:
            classifier (PersistenClassifier): A classifier object
    """
    print("Beggining to fit model")
    f,c = data[:,1:], data[:,0]
    classifier.fit(f,c)
    print("Finished fitting model")


def model(classifier, data):
    """ Tests a model
    Runs k folds validation on a data set
    Does not persist results, prints score
    Args:
        classifier (PersistenClassifier): A classifier object
    """
    print("Beggining to test model")
    train, test = cross_validation.train_test_split(data, test_size=.30)
    f,c = train[:,1:], train[:,0]
    classifier.fit(f,c,False)
    print("Score: " + classifier.score(f,c))
    print("Finished testing model")


def predict(classifier, data):
    """Predict the results of given data set
    Args:
        classifier (PersistentClassifier): our classifier
        data (np.array): our data set
    """
    print("Beggining to classify data")
    results = classifier.predict(data)
    results = pd.DataFrame(results)
    results.index += 1
    results.to_csv("out/results.csv", header=["Label"], index=True, index_label=["ImageId"])
    print("Finished classifying data")


def load_args():
    """ Load our config file
    Returns:
         (Dict): our arguments
    """
    parser = argparse.ArgumentParser(description="Classify and predict digits using the mnist dataset")
    parser.add_argument('mode', help='the mode to run in: fit, model or predict')
    parser.add_argument('--algo', help='which algorithm to use: RandomForest, KNN')
    return parser.parse_args()

def load_config():
    """ Load our config file
    Returns:
         ConfigParser
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

if __name__ == '__main__':
    """
        Load our arguments, configs and data and run the appropriate method
    """
    args, config = load_args(), load_config()

    if args.algo == None:
        algo = config['DEFAULT']['algo']

    classifier = models.load_classifier(algo, config[algo])

    if args.mode == 'fit':
        data = np.loadtxt(config[algo]['training_file'], dtype=np.int16, delimiter=',', skiprows=1)
        fit(classifier, data)
    elif args.mode == 'model':
        data = np.loadtxt(config[algo]['training_file'], dtype=np.int16, delimiter=',', skiprows=1)
        model(classifier, data)
    elif args.mode == 'predict':
        data = np.loadtxt(config[algo]['predict_file'], dtype=np.int16, delimiter=',', skiprows=1)
        predict(classifier, data)
    else:
        print('Please pass in train, model or predict.')


