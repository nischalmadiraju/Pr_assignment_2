import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import NearestNeighbors


def edge_detection(img):    # Edge detection for Fourier transform
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=200, threshold2=220)  # Canny Edge Detection
    return edges


def fourier_transform(img):
    # fourier transform and shift to center
    fourier_transform_shifted = np.fft.fftshift(np.fft.fft2(np.float32(img), axes=(0, 1)))
    magnitude_spectrum = np.log(np.abs(fourier_transform_shifted)) / 20
    descriptors = magnitude_spectrum
    return descriptors


def create_dictionary(X_train, k_fold): # Create a dictionary of keypoints, the best 10 keypoints for each image
    if k_fold == 0:
        dictionary = np.zeros((1270, 128), dtype=int)
    else:
        dictionary = np.zeros((1140, 128), dtype=int)

    sift = cv2.SIFT_create(10)
    idx = 0
    for image in X_train:
        image = np.reshape(image, (500, 500))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(image, None)
        for i in range(10):
            dictionary[idx] = descriptors[i]
            idx += 1

    return dictionary


def calculate_feature_vectors(X_train, X_test):  # Create SIFT feature vectors by comparing keypoints to the dictionary
    trainDataFeatures = []
    sift = cv2.SIFT_create()
    for image in X_train:
        image = np.reshape(image, (500, 500))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(image, None)

        dist = distance.cdist(descriptors, dictionary, 'euclidean')

        closest = np.argmin(dist, axis=1)

        imageFeatures = np.zeros(1700)
        for id_assign in closest:
            imageFeatures[id_assign] += 1

        trainDataFeatures.append(imageFeatures)

    testDataFeatures = []
    for image in X_test:
        image = np.reshape(image, (500, 500))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        keypoints, descriptors = sift.detectAndCompute(image, None)

        dist = distance.cdist(descriptors, dictionary, 'euclidean')

        closest = np.argmin(dist, axis=1)

        imageFeatures = np.zeros(1700)
        for id_assign in closest:
            imageFeatures[id_assign] += 1

        testDataFeatures.append(imageFeatures)

    return trainDataFeatures, testDataFeatures


def do_classification(X_train, y_train, X_test, y_test, featureExtraction, model): # Use the classification algorithms
    accuracy = 0
    if model == 0:
        if featureExtraction == 0:    # No feature extraction
            KNN = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=9, p=1)
        elif featureExtraction == 1:  # SIFT
            KNN = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=5, p=2)
        elif featureExtraction == 2:  # Fourier Transform
            KNN = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3, p=1)

        KNN.fit(X_train, y_train)
        KNNPredictions = KNN.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, KNNPredictions)

    elif model == 1:
        if featureExtraction == 0:    # No feature extraction
            randomForest = RandomForestClassifier(max_depth=100, min_samples_leaf=2, min_samples_split=4, n_estimators=50)
        elif featureExtraction == 1:  # SIFT
            randomForest = RandomForestClassifier(max_depth=50, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
        elif featureExtraction == 2:  # Fourier Transform
            randomForest = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=10)

        randomForest.fit(X_train, y_train)
        randomForestPredictions = randomForest.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, randomForestPredictions)

    elif model == 2:
        if featureExtraction == 0:    # No feature extraction
            logisticRegression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, C=0.5, tol=0.01)
        elif featureExtraction == 1:  # SIFT
            logisticRegression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, C=1.0, tol=0.01)
        elif featureExtraction == 2:  # Fourier Transform
            logisticRegression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, C=0.5, tol=0.01)

        logisticRegression.fit(X_train, y_train)
        logisticRegressionPredictions = logisticRegression.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, logisticRegressionPredictions)

    return accuracy


def do_grid_search(X_train, y_train):
    KNN_hyperparameter_space = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                'p': [1, 2, 3]}

    RF_hyperparameter_space = {'n_estimators': [10, 50, 100],
                               'criterion': ["gini"],
                               'max_depth': [10, 50, 100],
                               'min_samples_split': [2, 4, 6],
                               'min_samples_leaf': [1, 2, 3],
                               'bootstrap': [True]}

    LR_hyperparameter_space = {'tol': [0.01, 0.001],
                               'C': [0.5, 1.0, 1.5]}

    KNN = KNeighborsClassifier(n_jobs=4)
    KNNGridSearch = GridSearchCV(KNN, param_grid=KNN_hyperparameter_space, scoring='accuracy', n_jobs=4, cv=3)
    KNNGridSearch.fit(X_train, y_train)
    print("Optimal KNN hyperparameter combination:", KNNGridSearch.best_params_)
    print("Mean cross-validated training accuracy score:", KNNGridSearch.best_score_)

    randomForest = RandomForestClassifier(n_jobs=4)
    RFGridSearch = GridSearchCV(randomForest, param_grid=RF_hyperparameter_space, scoring='accuracy', cv=3, n_jobs=4)
    RFGridSearch.fit(X_train, y_train)
    print("Optimal Random Forest hyperparameter combination:", RFGridSearch.best_params_)
    print("Mean cross-validated training accuracy score:", RFGridSearch.best_score_)

    logisticRegression = LogisticRegression(max_iter=10000, solver='lbfgs', penalty='l2', n_jobs=4)
    LRGridSearch = GridSearchCV(logisticRegression, param_grid=LR_hyperparameter_space, scoring='accuracy', cv=3, n_jobs=4)
    LRGridSearch.fit(X_train, y_train)
    print("Optimal Logistic Regression hyperparameter combination:", LRGridSearch.best_params_)
    print("Mean cross-validated training accuracy score:", LRGridSearch.best_score_)

    return


if __name__ == "__main__":
    directory = os.fsencode('Data/BigCats')
    dataImages = np.zeros((170, 250000), dtype=int)
    dataLabels = []
    fourierFeatures = np.zeros((170, 12800), dtype=int)

    clustering = 0  # 1 for clustering, 0 for no clustering
    k_fold = 0  # 1 for k-fold cross-validation, 0 for train-test-split
    featureExtraction = 0  # 0 for no feature extraction, 1 for SIFT, 2 for Fourier Transform
    gridSearch = 0  # 1 for grid search, 0 for no grid search. Only works when k_fold is 0

    silhouette = 0
    idx = 0

    # Preprocessing
    for label in os.listdir(directory):
        subdirectory = os.path.join(directory, label)
        for file in os.listdir(subdirectory):
            filePath = os.path.join(subdirectory, file)
            fileName = os.fsdecode(filePath)
            if fileName.endswith(".jpg") or fileName.endswith(".jpeg"):  # Only use the images
                image = cv2.imread(fileName)

                if clustering == 1:
                    displayImage = cv2.resize(image, (50, 50))  # dimension reduction, otherwise silhouette score
                    clusterImage = displayImage.reshape(-1, 3)  # takes too long

                    #neigh = NearestNeighbors(n_neighbors=10)    # This can be uncommented to determine the value of eps
                    #nbrs = neigh.fit(clusterImage)
                    #distances, indices = nbrs.kneighbors(clusterImage)
                    #distances = np.sort(distances, axis=0)
                    #distances = distances[:, 1]
                    #plt.plot(distances)
                    #plt.show()

                    db = DBSCAN(eps=4.5, min_samples=10).fit(clusterImage[:, :2])

                    # plt.imshow(np.uint8(db.labels_.reshape(displayImage.shape[:2]))) # This can be uncommented to see
                    # plt.show()                                                       # the clustered image

                    silhouette += silhouette_score(clusterImage, db.labels_)

                image = cv2.resize(image, (500, 500))  # Resize the image to 500x500
                dataLabels.append(os.fsdecode(label))  # Save the labels
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Change the image to grayscale

                if featureExtraction == 2:  # For Fourier transform, the feature extraction happens here
                    img_edge = edge_detection(img=gray)
                    descriptors = fourier_transform(img=img_edge)
                    descriptors = descriptors.flatten()

                    for i in range(12800):
                        fourierFeatures[idx, i] = descriptors[i]

                else:
                    gray = gray.flatten()
                    for i in range(250000):
                        dataImages[idx, i] = gray[i]

                idx += 1

            else:
                continue

    if clustering == 1:
        print("Mean silhouette score:", silhouette / 170)

    if k_fold == 1:  # k-fold cross validation
        dataLabels = np.array(dataLabels)
        k_folds = KFold(n_splits=3, shuffle=True)
        KNNAccuracy = 0
        randomForestAccuracy = 0
        logisticRegressionAccuracy = 0
        if featureExtraction == 2:
            for train_index, test_index in k_folds.split(fourierFeatures, dataLabels):
                X_train, X_test = fourierFeatures[train_index], fourierFeatures[test_index]
                y_train, y_test = dataLabels[train_index], dataLabels[test_index]

                if featureExtraction == 1:
                    #  Create keypoint dictionary (10 keypoints per image) using only the training data
                    dictionary = create_dictionary(X_train, k_fold)

                    # Calculate feature vectors for the training and testing data
                    X_train, X_test = calculate_feature_vectors(X_train, X_test)

                # Classification
                KNNAccuracy += do_classification(X_train, y_train, X_test, y_test, featureExtraction, 0)
                randomForestAccuracy += do_classification(X_train, y_train, X_test, y_test, featureExtraction, 1)
                logisticRegressionAccuracy += do_classification(X_train, y_train, X_test, y_test, featureExtraction, 2)
        else:
            for train_index, test_index in k_folds.split(dataImages, dataLabels):
                X_train, X_test = dataImages[train_index], dataImages[test_index]
                y_train, y_test = dataLabels[train_index], dataLabels[test_index]

                if featureExtraction == 1:
                    #  Create keypoint dictionary (10 keypoints per image) using only the training data
                    dictionary = create_dictionary(X_train, k_fold)

                    # Calculate feature vectors for the training and testing data
                    X_train, X_test = calculate_feature_vectors(X_train, X_test)

                # Classification
                KNNAccuracy += do_classification(X_train, y_train, X_test, y_test, featureExtraction, 0)
                randomForestAccuracy += do_classification(X_train, y_train, X_test, y_test, featureExtraction, 1)
                logisticRegressionAccuracy += do_classification(X_train, y_train, X_test, y_test, featureExtraction, 2)

        print("Mean accuracy for KNN:", KNNAccuracy / 3)
        print("Mean accuracy for Random Forest:", randomForestAccuracy / 3)
        print("Mean accuracy for Logistic Regression:", logisticRegressionAccuracy / 3)

    else:  # train-test split
        if featureExtraction == 2:
            X_train, X_test, y_train, y_test = train_test_split(fourierFeatures, dataLabels)
        else:
            X_train, X_test, y_train, y_test = train_test_split(dataImages, dataLabels)

        if featureExtraction == 1:
            dictionary = create_dictionary(X_train, k_fold)  # Create keypoint dictionary using only the training data
            X_train, X_test = calculate_feature_vectors(X_train, X_test)  # Calculate feature vectors

        if gridSearch == 1:
            X_train = np.r_[X_train, X_test]
            y_train = np.r_[y_train, y_test]
            do_grid_search(X_train, y_train)

        # Classification
        print("Accuracy for KNN:", do_classification(X_train, y_train, X_test, y_test, featureExtraction, 0))
        print("Accuracy for Random Forest:", do_classification(X_train, y_train, X_test, y_test, featureExtraction, 1))
        print("Accuracy for Logistic Regression:", do_classification(X_train, y_train, X_test, y_test, featureExtraction, 2))
