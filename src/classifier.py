import pickle
import os
import pandas as pd
import numpy as np

from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import align
import rep

def train(labelsPath, repsPath, outDir):
    labels = []
    file = open(labelsPath, "r")
    for line in file:
        labels.append(line.rstrip("\n"))
    file.close()

    embeddings = pd.read_csv(repsPath, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(embeddings, labelsNum)

    with open(os.path.join(outDir, "classifier.pkl"), 'w') as f:
        pickle.dump((le, clf), f)

def infer(reps, labelEncoder, classifier):
    results = []
    for rep in reps:
        repValues = rep[1].reshape(1, -1)
        predictions = classifier.predict_proba(repValues).ravel()
        maxI = np.argmax(predictions)
        person = labelEncoder.inverse_transform(maxI)
        confidence = predictions[maxI]
        results.append((person.decode("utf-8"), confidence))
    return results
