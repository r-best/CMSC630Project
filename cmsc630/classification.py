import random
import logging
import numpy as np


def feature_extraction(images, save_to='dataset.csv'):
    """Takes in an array of Image objects and returns a dataset constructed of
    features representing them, detailed below
    
    1-4: number of object pixels (for each color)
    5. standard deviation of grayscale histogram
    6. The highest peak of the grayscale histogram
    7. The lowest valley of the grayscale histogram

    Arguments:
        images (array-like): A list of Image objects to extract features from
        save_to (string): Filepath to save the dataset to
    
    Returns:
        x (ndarray): The dataset, (nxfeatures)
        y (ndarray): The labels corresponding to the dataset samples, (,n)
    """
    num_images = len(images)
    logging.info(f"Extracting features from {num_images} images...")
    x = np.zeros((num_images, 7))
    y = np.zeros(num_images, dtype=np.int8)

    for i, image in enumerate(images):
        logging.info(f"Processing Image {i+1}/{num_images}...")
        y[i] =   0 if image.name.startswith('cyl')   \
            else 1 if image.name.startswith('inter') \
            else 2 if image.name.startswith('let')   \
            else 3 if image.name.startswith('mod')   \
            else 4 if image.name.startswith('para')  \
            else 5 if image.name.startswith('super') \
            else 6 if image.name.startswith('svar') else -1
        
        # Get number of object pixels in segmented color channels, which become features 0-3
        for color in [0,1,2,4]: # 3 is the color index for RGB so we skip that and use 4 (grayscale)
            uniques, counts = np.unique(image.getMatrix(color), return_counts=True)
            if len(uniques) > 2:
                image = image.otsu(color)
                uniques, counts = np.unique(image.getMatrix(color), return_counts=True)
            x[i,color if color is not 4 else 3] = counts[0]

            x[i,4] = np.std(image.getHistogram(4))

            x[i,5] = np.argmax(image.getHistogram(4))

            x[i,6] = np.argmin(image.getHistogram(4))

    # Save new dataset to file
    np.savetxt(save_to, np.concatenate([x,np.atleast_2d(y).T], axis=1), delimiter=',', fmt='%s')

    return x, y


def cross_validate(cv, x, y, k=1):
    """Takes in a dataset and a cross-validation value (cv) and performs cross-validation
    by splitting up the dataset into cv segments and iteratively using one as the test set
    and the rest as the training set, averaging the results at the end

    Arguments:
        cv (int): The number of folds, i.e. segments to divide the dataset into
        x (ndarray): The dataset, (nxfeatures)
        y (ndarray): The labels corresponding to the dataset samples, (,n)
        k (int): The value of k to pass along to the k-nearest-neighbors algorithm
    
    Returns:
        The averaged performance metrics across all folds (precision, recall, F1, accuracy)
    """
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    stepsize = int(len(x) / cv)
    metrics = np.zeros(4)
    for i in range(cv):
        logging.info(f"Cross-validation fold {i+1}")

        # Slice test set out of data
        test_indices = indices[i*stepsize:i*stepsize+stepsize]
        x_test = x[test_indices]
        y_test = y[test_indices]

        # Everything else is the training set
        x_train = np.copy(x)
        x_train = np.delete(x_train, test_indices, axis=0)
        y_train = np.copy(y)
        y_train = np.delete(y_train, test_indices, axis=0)

        metrics += evaluate(knn(x_test, x_train, y_train, k), y_test)
    metrics /= cv

    print(metrics)
    return metrics


def knn(x, x_train, y_train, k=1):
    """Takes in a training and test dataset (samplesxfeatures matrices) and a set of
    labels for the training set (samplesx1 matrix) and uses a k-nearest-neighbors
    algorithm to classify the test samples, returning the labels in the corresponding order

    Arguments:
        x (ndarray): An (n,features) test dataset
        x_train (ndarray): An (m,features) training dataset
        y_train (ndarray): A (,m) array of labels for the training data
        k (int): The nearest-neighbors number
    
    Returns:
        An (,n) ndarray of predicted labels for the given test set
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
    """Takes in an array of predicted labels and a corresponding array of true gold
    standard labels and computes performance metrics from them

    Arguments:
        labels (array-like): Predicted labels
        gold (array-like): True labels
    
    Returns:
        Micro-averaged precision, recall, F1, and accuracy metrics
    """
    num_labels = np.max([np.max(labels),np.max(gold)])+1

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
    precision_micro = np.sum(TP) / np.sum(TP+FP)
    recall_micro = np.sum(TP) / np.sum(TP+FN)
    F1_micro = 2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro))
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    return precision_micro, recall_micro, F1_micro, accuracy
