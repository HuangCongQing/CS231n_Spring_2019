from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange

# KNN类名
class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass
    
    # 训练   classifier = KNearestNeighbor()  classifier.train(X_train, y_train)
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    # 预测
    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    # 2循环：用了两个循环的算法实现（L2距离）
    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0] # X_test)  X测试集
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) # 初始化L2距离 500x5000
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                # 使用numpy库对两个向量进行欧几里得距离计算
                # 首先在两重循环中将测试集和训练集各一行向量通过减法运算后，平方，然后求和，最后开根号，这样就完成了欧式距离计算
                # numpy.square（）平方函数返回一个新数组，该数组的元素值为源数组元素的平方。 源阵列保持不变
                dists[i,j] = np.sqrt(np.sum(np.square(X[i]) - self.X_train[j]))
                # pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists # (500, 5000)

    # 1循环：用了一个循环的算法实现（L2距离）# 使用了广播机制，省去了一个循环.
    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 使用了广播机制，省去了一个循环.
            # X[i] -> shape(D,)    self.X.train -> shape(num_train, D)
            dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis = 1))
            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    # 无循环：不用循环的算法实现（L2距离）
    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        d1 = np.sum(np.square(X), axis = 1, keepdims = True)
        d2 = - 2 * np.dot(X, self.X_train.T)
        d3 = np.sum(np.square(self.X_train.T), axis = 0, keepdims = True)
        
        assert(d1.shape == (num_test, 1))
        assert(d2.shape ==(num_test, num_train))
        assert(d3.shape == (1, num_train))
        dists = np.sqrt(d1 + d2 + d3)
        # 参考： https://blog.csdn.net/qq_33445835/article/details/104414989

        # pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    # 预测标签
    # 若k个近邻投票产生多个结果时，选用标签较小的那个，此要求在本次示例中已完成，
    # np.bicount()会返回0到数组中最大值出现的次数数组，np.argmax会返回次数数组中次数最多的索引（如果有多个，则返回最前面的），即标签较小的。
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # pass
            # dists是一个确定的矩阵：：纵轴1维测试集，横轴2维训练集(知道label)
            # 得到label ，即y_train
            # argsort()是将X中的元素(按照列)从小到大排序后，提取对应的索引index，然后输出
            # 沿着列向右(每行)的元素进行排序
            closest_y = self.y_train[np.argsort(dists[i])[0: k]].astype(np.int32)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # pass
            # np.bicount()会返回0到数组中最大值出现的次数数组，np.argmax会返回次数数组中次数最多的索引（如果有多个，则返回最前面的），即标签较小的。
            y_pred[i] = np.argmax(np.bincount(closest_y)) # 出现次数最多的label

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
