import random
import logging
import numpy as np
from sklearn import metrics


def feature_extraction(images):
    """
    1. number of black pixels
    2. 
    3. 
    4. 
    """
    x = np.zeros((len(images), 4))
    y = np.zeros(len(images), dtype=np.int8)

    for i, image in enumerate(images):
        y[i] =   0 if image.name.startswith('cyl')   \
            else 1 if image.name.startswith('inter') \
            else 2 if image.name.startswith('let')   \
            else 3 if image.name.startswith('mod')   \
            else 4 if image.name.startswith('para')  \
            else 5 if image.name.startswith('super') \
            else 6 if image.name.startswith('svar') else -1

        x[i,0] = np.unique(image.getMatrix(4), return_counts=True)[1][0]

    return x, y


def cross_validate(cv, x, y, k=1):
    """
    """
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    stepsize = int(len(x) / cv)
    metrics = np.zeros(4)
    for i in range(cv):
        x_train = x[indices[i*stepsize:(i+1)*stepsize]]
        y_train = y[indices[i*stepsize:(i+1)*stepsize]]
        for j in range(cv):
            if i == j: continue
            test_start = j*stepsize
            metrics += evaluate(
                knn(x[indices[test_start:test_start+stepsize]], x_train, y_train, k),
                y[indices[test_start:test_start+stepsize]]
            )
    metrics /= cv*(cv-1)
    print(metrics)


def knn(x, x_train, y_train, k=1):
    """
    """
    y_pred = np.zeros(len(x), dtype=np.int8)
    for i, sample in enumerate(x):
        # Calculate distance from this sample to every training sample
        dist = [np.linalg.norm(sample-train) for train in x_train]

        # Find the k nearest training samples
        k_nearest_labels = []
        for j in range(k):
            closest = np.argmin(dist)
            k_nearest_labels.append(y_train[closest])
            dist.pop(closest)
        
        # This sample's label the one the appears most frequently in
        # the k nearest, or the first nearest if all appear equally
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred[i] = labels[np.argmax(counts)]
    return y_pred


def evaluate(labels, gold):
    """
    """
    num_labels = np.max(gold)+1

    # Compute confusion matrix
    conf_matrix = np.zeros((num_labels, num_labels), dtype=np.int)
    for i, _ in enumerate(labels):
        conf_matrix[labels[i], gold[i]] += 1
    
    # Compute metrics
    TP = np.zeros(num_labels, dtype=np.int)
    FP = np.zeros(num_labels, dtype=np.int)
    TN = np.zeros(num_labels, dtype=np.int)
    FN = np.zeros(num_labels, dtype=np.int)
    for i in range(num_labels):
        TP[i] = conf_matrix[i, i]
        FP[i] = np.sum(conf_matrix[:i,i]) + np.sum(conf_matrix[i+1:,i])
        TN[i] = np.sum(conf_matrix[:i,:i]) + np.sum(conf_matrix[i+1:,:i]) + \
                    np.sum(conf_matrix[:i,i+1:]) + np.sum(conf_matrix[i+1:,i+1:])
        FN[i] = np.sum(conf_matrix[i,:i]) + np.sum(conf_matrix[i,i+1:])
    print(np.sum(TP+TN+FP+FN), np.sum(conf_matrix))
    precision_micro = np.sum(TP) / np.sum(TP+FP)
    recall_micro = np.sum(TP) / np.sum(TP+FN)
    F1_micro = 2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro))
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    return precision_micro, recall_micro, F1_micro, accuracy
