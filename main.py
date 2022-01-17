import numpy as np


class Classification:
    # /----------------------------------------------------------------------------------------------------\#
    # function : buildData =>
    # Create 4 arrays from csv file passing to function
    # (Trainning features and Test features is 2D array- Training labels and test labels is 2D array)
    # feature Cols : number of feature columns in file
    # test ratio : how percent of data is used to test
    def buildData(self, filename, featureCols, testRatio):
        f = open(filename)
        data = np.loadtxt(fname=f, delimiter=',')
        ##       example : data =   [[ 40. 200. 250. 140. 104. 111. 111. 114. 127.  0.]
        ##                          [120. 200. 250. 139. 124. 125. 116. 113. 130.   0.]
        ##                          [ 40. 450. 250. 185. 150. 134. 146. 141. 146.   1.]
        ##                          [120. 450. 250. 209. 177. 168. 200. 168. 172.   0.]]
        ##                         X:|------------------------------------------|y:|--|
        # randomly confuse the all data
        np.random.shuffle(data)
        # X if fearures : all columns except last column
        # (array indices : row => all and column => all except last column)
        Features = data[:, :featureCols]
        ##      Features =[[ 40. 325. 150. 183. 131. 123. 133. 117. 114.]
        ##                  [ 80. 450. 350. 182. 180. 184. 192. 175. 191.]
        ##                  [ 40. 450. 250. 185. 150. 134. 146. 141. 146.]
        ##                  [120. 450. 250. 209. 177. 168. 200. 168. 172.]]

        # Y is label : only last column is label
        # (array indices : row => all and column => last column)
        Labels = data[:, featureCols]
        ##        Labels =[0. 1. 0. 0.]
        # number of lines
        Lines = Labels.size
        # print ("\n| Imported " + str(Lines) + " lines.\t\t\t|")
        # split is a point that Training data seprates test data
        split = int((1 - testRatio) * Lines)
        # X is 2D Array and y is 1D Array
        # X is Features and y is labels
        # Training Features (array indices : row => 0 to split point and column => all)
        Training_Features = Features[0:split, :]
        # Test Features (array indices : row => split point to end of row and column => all)
        Test_Features = Features[split:, :]
        # Training labels (array indices : 0 to split point)
        Training_Labels = Labels[0:split]
        # Test labels (array indices : split point to end of the array)
        Test_Labels = Labels[split:]
        # this items for kmeans clustring : include all labels and all features
        All_Features = Features[0:Lines, :]
        All_Labels = Labels[:]
        # function return Training and Test features(X) with labels(y)
        return Training_Features, Training_Labels, Test_Features, Test_Labels, All_Features, All_Labels

    # /----------------------------------------------------------------------------------------------------\#
    def svm_Classifiction(self, c, g, Training_Features, Training_Labels, Test_Features, Test_Labels):
        # import classifier
        from sklearn import svm
        from sklearn.metrics import accuracy_score
        # C : low c => Smooth or high c => more accurate boundary
        # c = 1
        # Gamma : low g => less data or high g => more data impact boundary
        # g = 0.1
        # select the classifier
        # Kernel type is RBF
        clf = svm.SVC(kernel='rbf', C=c, gamma=g)
        # classifier fit the boundary to training data
        clf.fit(Training_Features, Training_Labels)
        # classifier predict the labels for training data
        pred = clf.predict(Training_Features)
        # and accuracy is calculate for training data
        Training_Accuracy = accuracy_score(pred, Training_Labels)
        # classifier predict the labels for test data
        pred = clf.predict(Test_Features)
        # and accuracy is calculate for test data
        Test_Accuracy = accuracy_score(pred, Test_Labels)
        # function return training and test accuracy with c and gamma parameters
        return Training_Accuracy, Test_Accuracy, c, g

    # /----------------------------------------------------------------------------------------------------\#
    def naive_bayes_Classifiction(self, Training_Features, Training_Labels, Test_Features, Test_Labels):
        # import classifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        # select classifier
        clf = GaussianNB()
        # classifier fit the boundary to training data
        clf.fit(Training_Features, Training_Labels)
        # classifier predict the labels for training data
        pred = clf.predict(Training_Features)
        # and accuracy is calculate for training data
        Training_Accuracy = accuracy_score(pred, Training_Labels)
        # classifier predict the labels for test data
        pred = clf.predict(Test_Features)
        # and accuracy is calculate for test data
        Test_Accuracy = accuracy_score(pred, Test_Labels)
        # function return training and test accuracy
        return Training_Accuracy, Test_Accuracy

    # /----------------------------------------------------------------------------------------------------\#
    def knn_Classifiction(self, k, Training_Features, Training_Labels, Test_Features, Test_Labels):
        # import classifier
        from sklearn import neighbors
        from sklearn.metrics import accuracy_score
        # the number of neighbors impact
        # k = 7
        # select classifier
        clf = neighbors.KNeighborsClassifier(k, weights='uniform')
        # classifier fit the boundaris to training data
        clf.fit(Training_Features, Training_Labels)
        # classifier predict the labels for training data
        pred = clf.predict(Training_Features)
        # and accuracy is calculate for training data
        Training_Accuracy = accuracy_score(pred, Training_Labels)
        # classifier predict the labels for test data
        pred = clf.predict(Test_Features)
        # and accuracy is calculate for test data
        Test_Accuracy = accuracy_score(pred, Test_Labels)
        # function return training and test accuracy with k
        return Training_Accuracy, Test_Accuracy, k

    # /----------------------------------------------------------------------------------------------------\#
    def Decision_TREE_Classifiction(self, d, Training_Features, Training_Labels, Test_Features, Test_Labels):
        # import classifier
        from sklearn import tree
        from sklearn.metrics import accuracy_score
        # max depth of tree
        # d = 7
        # select classifier
        clf = tree.DecisionTreeClassifier(max_depth=d)
        # classifier fit the boundaris to training data
        clf.fit(Training_Features, Training_Labels)
        # classifier predict the labels for training data
        pred = clf.predict(Training_Features)
        # and accuracy is calculate for training data
        Training_Accuracy = accuracy_score(pred, Training_Labels)
        # classifier predict the labels for test data
        pred = clf.predict(Test_Features)
        # and accuracy is calculate for test data
        Test_Accuracy = accuracy_score(pred, Test_Labels)
        # function return training and test accuracy with depth
        return Training_Accuracy, Test_Accuracy, d

    # /----------------------------------------------------------------------------------------------------\#
    def kmeans_Clustering(self, k, All_Features, All_Labels):
        # import cluster
        from sklearn.cluster import KMeans
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        from sklearn import decomposition
        pca = decomposition.PCA(n_components=2)
        pca.fit(All_Features)
        X = pca.transform(All_Features)
        # input("Press Any Key To Show Plots . . . ")
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
        # number of clusters
        k = 3
        # select kmeans cluster with k
        cls = KMeans(k)
        # cluster fit the data
        cls.fit(X)
        # centroid of clusters
        centroids = cls.cluster_centers_
        # labels of each cluster
        labels = cls.labels_
        # print("centroids : \n", centroids , "\n Number of Centroids : " , len(centroids))
        # print("labels : \n", labels , "\n Number of labels : ", len(labels))
        colors = ["g.", "r.", "c.", "y."]
        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
        plt.show()

        # predict label for clusters
        pred = cls.predict(X)
        # and accuracy is calculate for labels
        accuracy = accuracy_score(pred, All_Labels)
        # function return accuracy with k
        return accuracy, k
    # /----------------------------------------------------------------------------------------------------\#
