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

from sklearn.neighbors import NearestNeighbors

from Feature_extraction_pipeline import edge_detection, fourier_transform


def create_dictionary(X_train):
    dictionary = np.zeros((1700, 128), dtype=int)
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
    
    
def calculate_feature_vectors(X_train, X_test):
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
    
    
def do_classification(X_train, y_train, X_test, y_test, model):
    accuracy = 0
    if model == 0:
        KNN = KNeighborsClassifier()
        KNN.fit(X_train, y_train)
        KNNPredictions = KNN.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, KNNPredictions)
    elif model == 1:
        randomForest = RandomForestClassifier(n_estimators=100)
        randomForest.fit(X_train, y_train)
        randomForestPredictions = randomForest.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, randomForestPredictions)
    elif model == 2:  # Add logistic regression here
        logisticRegression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
        logisticRegression.fit(X_train, y_train)
        logisticRegressionPredictions = logisticRegression.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, logisticRegressionPredictions)
        
    return accuracy
    

if __name__ == "__main__":
    directory = os.fsencode('Data/BigCats')
    dataImages = np.zeros((170, 250000), dtype=int)
    dataLabels = []
    fourierFeatures = np.zeros((170, 12800), dtype=int)

    clustering = 0          # 1 for clustering, 0 for no clustering
    k_fold = 1              # 1 for k-fold cross-validation, 0 for train-test-split
    featureExtraction = 1   # 0 for no feature extraction, 1 for SIFT, 2 for Fourier Transform

    silhouette = 1
    idx = 0

    #  Preprocessing
    for label in os.listdir(directory):
        subdirectory = os.path.join(directory, label)
        for file in os.listdir(subdirectory):
            filePath = os.path.join(subdirectory, file)
            fileName = os.fsdecode(filePath)
            if fileName.endswith(".jpg") or fileName.endswith(".jpeg"):  # Only use the images
                image = cv2.imread(fileName)

                if clustering == 1:
                    displayImage = cv2.resize(image, (50, 50))  # dimension reduction, otherwise silhouette score
                    #clusterImage = clusterImage.flatten()       # takes too long
                    clusterImage = displayImage.reshape(-1, 3)

                    #neigh = NearestNeighbors(n_neighbors=3)    # This can be uncommented to determine the value of eps
                    #nbrs = neigh.fit(clusterImage)
                    #distances, indices = nbrs.kneighbors(clusterImage)
                    #distances = np.sort(distances, axis=0)
                    #distances = distances[:, 1]
                    #plt.plot(distances)
                    #plt.show()

                    db = DBSCAN(eps=4.5, min_samples=10).fit(clusterImage[:, :2])

                    #plt.imshow(np.uint8(db.labels_.reshape(displayImage.shape[:2]))) # This can be uncommented to see
                    #plt.show()                                                       # the clustered image

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
        print("Mean silhouette score:", silhouette/170)

    if k_fold == 1:
        dataLabels = np.array(dataLabels)
        k_folds = KFold(n_splits=3, shuffle=True)
        KNNAccuracy = 0
        randomForestAccuracy = 0
        logisticRegressionAccuracy = 0
        for train_index, test_index in k_folds.split(dataImages, dataLabels):
            X_train, X_test = dataImages[train_index], dataImages[test_index]
            y_train, y_test = dataLabels[train_index], dataLabels[test_index]

            #  Create keypoint dictionary (10 keypoints per image) using only the training data
            dictionary = create_dictionary(X_train)

            # Calculate feature vectors for the training and testing data
            X_train, X_test = calculate_feature_vectors(X_train, X_test)

            # Classification
            KNNAccuracy += do_classification(X_train, y_train, X_test, y_test, 0)
            randomForestAccuracy += do_classification(X_train, y_train, X_test, y_test, 1)
            logisticRegressionAccuracy += do_classification(X_train, y_train, X_test, y_test, 2)

            print("Mean accuracy for KNN:", KNNAccuracy/3)
            print("Mean accuracy for Random Forest:", randomForestAccuracy/3)
            print("Mean accuracy for Logistic Regression:", logisticRegressionAccuracy/3)
                
    else:
        X_train, X_test, y_train, y_test = train_test_split(dataImages, dataLabels, random_state=9)

        if featureExtraction == 1:
            dictionary = create_dictionary(X_train)  # Create keypoint dictionary using only the training data
            X_train, X_test = calculate_feature_vectors(X_train, X_test) # Calculate feature vectors

        # Classification
        print("Accuracy for KNN:", do_classification(X_train, y_train, X_test, y_test, 0))
        print("Accuracy for Random Forest:", do_classification(X_train, y_train, X_test, y_test, 1))
        print("Accuracy for Logistic Regression:", do_classification(X_train, y_train, X_test, y_test, 2))
