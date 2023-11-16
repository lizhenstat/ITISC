# code-from：https://github.com/omadson/fuzzy-c-means/blob/master/fcmeans/fcm.py
import argparse
import imageio
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist # l2: default
import matplotlib.pyplot as plt
import pandas as pd

class FCM:
    """Fuzzy C-means
    
    Parameters
    ----------
    n_clusters: int, optional (default=10)
        The number of clusters to form as well as the number of
        centroids to generate

    max_iter: int, optional (default=150)
        Hard limit on iterations within solver.

    m: float, optional (default=2.0)
        Exponent for the fuzzy partition matrix, specified as a
        scalar greater than 1.0. This option controls the amount of
        fuzzy overlap between clusters, with larger values indicating
        a greater degree of overlap.


    error: float, optional (default=1e-5)
        Tolerance for stopping criterion.

    random_state: int, optional (default=42)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    Attributes
    ----------
    n_samples: int
        Number of examples in the data set

    n_features: int
        Number of features in samples of the data set

    u: array, shape = [n_samples, n_clusters]
        Fuzzy partition array, returned as an array with n_samples rows
        and n_clusters columns. Element u[i,j] indicates the degree of
        membership of the jth data point in the ith cluster. For a given
        data point, the sum of the membership values for all clusters is one.

    centers: array, shape = [n_class-1, n_SV]
        Final cluster centers, returned as an array with n_clusters rows
        containing the coordinates of each cluster center. The number of
        columns in centers is equal to the dimensionality of the data being
        clustered.

    r: 
    Container for the Mersenne Twister pseudo-random number generator.
    
    Methods
    -------
    fit(X)
        fit the data

    _predict(X)
        use fitted model and output cluster memberships

    predict(X)
        use fitted model and output 1 cluster for each sample

    References
    ----------
    .. [1] `Pattern Recognition with Fuzzy Objective Function Algorithms
        <https://doi.org/10.1007/978-1-4757-0450-1>`_
    .. [2] `FCM: The fuzzy c-means clustering algorithm
        <https://doi.org/10.1016/0098-3004(84)90020-7>`_

    """
    def __init__(self, args):
        self.u, self.centers = None, None
        self.n_clusters = args.C
        self.max_iter = args.max_iter
        self.m = args.m
        self.error = args.error
        self.random_state = args.fcm_random_state
        assert self.m > 1

        # save gif
        self.gif = args.gif
        self.gif_gap = args.gif_gap
        self.plot_boundary = args.plot_boundary
        self.prob = args.prob
        self.image_list = []

    def fit(self, X):
        """Compute fuzzy C-means clustering.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training instances to cluster.
        """
        self.n_samples = X.shape[0]
        r = np.random.RandomState(self.random_state)
        self.u = r.rand(self.n_samples,self.n_clusters)
        # u matrix initialization: random 
        self.u = self.u / np.tile(self.u.sum(axis=1)[np.newaxis].T, self.n_clusters)

        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            self.centers = self.next_centers(X)
            self.u = self._predict(X)
            if norm(self.u - u_old) < self.error: # Stopping rule
               break
        
            # save intermediate result
            if self.gif:
               if iteration % self.gif_gap == 0: 
                  image, fig= self.get_gif(X, iteration, plot_boundary=self.plot_boundary)
                  fig.clf()
                  self.image_list = self.image_list + [image]

        if self.gif:
            imageio.mimsave('./gif/FCM_m={}.gif'.format(self.m), self.image_list, fps=1)
        print('FCM break at iteration:{}'.format(iteration))

    def next_centers(self, X):
        """Update cluster centers"""
        um = self.u ** self.m 
        return (X.T @ um / np.sum(um, axis=0)).T 
    
    ### U_update
    def _predict(self, X):
        """ 
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        u: array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.

        """
        power = float(2 / (self.m - 1))
        temp = cdist(X, self.centers) ** power 
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        res = 1 / denominator_.sum(2)
        return res

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.

        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        u = self._predict(X)
        return np.argmax(u, axis=-1)
    

    def predict_new(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        u = self._predict(X)
        return u

    def get_gif(self, X, iter, plot_boundary=False):
        fig, ax = plt.subplots()
        colors_tmp = ['g', 'r', 'b', 'c', 'm', 'k','y','cyan','pink','gray']
        point_size = 3
        alpha = 0.8
        color_tmp = colors_tmp[:self.n_clusters]
        pred = self.predict(X)

        if  plot_boundary:
            rtol = 1e-2
            mesh_grid_pred = get_meshgrid(X)
            mesh_grid_U_pred = self.predict_new(mesh_grid_pred)
            cls_ind_list = []
            for ind in range(self.n_clusters):
                cls_ind_tmp = get_boundary_index(mesh_grid_U_pred, ind, self.prob, rtol=rtol)
                cls_ind_list = cls_ind_list + [cls_ind_tmp]
        
        ax.plot(self.centers[:,0],  self.centers[:,1], "*" , markersize = 10, c = 'blue') # DA centers

        for ind in range(self.n_clusters):
            ax.scatter(X[pred==ind,0], X[pred==ind,1], s=5, alpha=0.5, c=colors_tmp[ind]) # 数据点
            if plot_boundary:
               ax.scatter(mesh_grid_pred[cls_ind_list[ind],0],mesh_grid_pred[cls_ind_list[ind],1], c = colors_tmp[ind], s=point_size, alpha=alpha)

        ax.set_title('iter={}, m={}'.format(iter, self.m))
        fig.canvas.draw()      
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image, fig


if __name__ == '__main__':    
    from hyperparam import args
    from utility_code import get_data, generate_data_ind, get_meshgrid, get_boundary_index
    
    # data
    args.T2 = 0.7
    args.C = 3
    args.R = 1
    args = get_data(args)
    data, labels, true_centers, data_mean, dataDistDesInd = generate_data_ind(args)
    
    # model
    model = FCM(args)
    model.fit(data)
    centers = model.centers
    pred = model.predict(data)
    print('centers\n', centers)
    
    if not args.gif:
        plt.scatter(data[:,0],data[:,1], c=pred, s=10)
        plt.scatter(centers[:,0],centers[:,1], c='k', s=30)
        plt.show()  



