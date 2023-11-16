# ITISC Alternative Optimization algorithm
import math
import random
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist # l2: default
from math import sqrt
import matplotlib.pyplot as plt
import imageio

class ITISC_AO:
    def __init__(self, args, initCenters=None):
        self.T1 = args.T1
        self.T2 = args.T2
        self.max_iter = args.max_iter
        assert self.T1 > 0
        assert self.T2 > 0
        self.T2_list = [] 
        self.u, self.centers = None, None
        self.n_clusters = args.C
        self.max_iter = args.max_iter
        self.error = args.error
        self.itisc_random_state = args.itisc_random_state
        self.args = args

        # intermediate step
        self.all_W = []
        self.all_centers = []
        self.all_pred = []
        self.convergeOrNot = None 
        
        # save gif
        self.gif = args.gif
        self.gif_gap = args.gif_gap
        self.prob = args.prob
        self.image_list = []
        self.initCenters = initCenters
        self.init_method = args.init_method
        self.plot_boundary = args.plot_boundary # isoprobability curve

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.r = np.random.RandomState(self.itisc_random_state)
        self.U = self.r.rand(self.n_samples,self.n_clusters)
        self.U = self.U / np.tile(self.U.sum(axis=1)[np.newaxis].T, self.n_clusters)
        self.W = self.r.rand(self.n_samples,1)
        self.W = self.W / np.sum(self.W) # normalize
        self.all_W = [self.W]
        
        for iteration in range(self.max_iter):
            if iteration == 0:
               self.centers = self.init_centers(X)
               self.all_centers = [self.centers]

            U_old = self.U.copy()
            centers_old = self.centers.copy()
            self.U = self.next_u(X)
            
            self.W = self.next_w(X)
            self.all_W = self.all_W + [self.W]

            self.centers = self.next_centers(X, iteration)
            self.all_centers = self.all_centers + [self.centers]
            self.all_pred = self.all_pred + [self.predict(X)]
            # Stopping rule
            if not np.isnan(self.centers).any():
               # if norm(self.U - U_old) < self.error:
               if norm(self.centers - centers_old) < self.error: # same as ITISC_R
                   print('Centers Y not changed, Break iteration={}'.format(iteration))
                   break
            else:
                print('Centers Y contains nan value, break at iteration={}'.format(iteration))
                break

            ## save intermediate step
            if self.gif:
                if iteration % self.gif_gap == 0: 
                    image, fig= self.get_gif(X, iteration, plot_boundary=self.plot_boundary)
                    fig.clf()
                    self.image_list = self.image_list + [image]
        if self.gif:
           imageio.mimsave('./gif/ITISC_AO_T1={}_T2={}.gif'.format(self.T1, self.T2), self.image_list, fps=1)  
 
        print('ITISC_AO break at iteration:{}, T1={:.2f}, T2={:.2f}'.format(iteration, self.T1, self.T2))
        print('iteration={}, centers_diff={}'.format(iteration, norm(self.centers - centers_old)))
        print('center norm={}'.format(norm(self.centers - centers_old)))
        self.convergeOrNot = norm(self.centers - centers_old) < self.error
        print('The ITISC_AO algorithm converge={}'.format(self.convergeOrNot))
        return self.T1, self.T2, self.convergeOrNot
      
    def init_centers(self, X):
        # provided init Centers: may come from another clustering algorithm
        if not self.initCenters is None: 
           initCenters = self.initCenters
        elif self.init_method == 'random': # random initialization from data range
            random.seed(self.args.itisc_random_state) 
            xmax = np.max(X, axis=0)
            xmin = np.min(X, axis=0)
            initCenters = []
            for c in range(self.n_clusters):
                tmp = [random.uniform(tmp_min, tmp_max) for tmp_min, tmp_max in zip(xmin, xmax)]
                initCenters.append(tmp)
            initCenters = np.array(initCenters)
        elif self.init_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.args.random_state).fit(X)
            initCenters = kmeans.cluster_centers_ 
        elif self.init_method == 'fcm':
            from fcm_github import FCM
            fcm = FCM(self.args)
            fcm.fit(X)
            initCenters = fcm.centers
        else:
            exit('Unrecognized init center method')
        return initCenters

    def next_centers(self, X, iteration):
        # based on updated u_{ij} and w_{i}
        # xi: observation
        # yj: current center
        # i : index for observation，total N
        # j : index for center，total C
        # U : [N,C] # numerator:[N,C] denominator:[N,1] --> update u_{ij} (公式1)
        # W : [N,1] # numerator:[N,1] denominator:[1,1] --> update w_{i}  (公式2)
        # X:  [N,R] # R: feature dimension
        W = self.W ** (1-self.T2) 
        U = self.U ** (1+self.T1)
        D = cdist(X, self.centers)**2 # [N,C] 
        
        ### xi overlap with cluster center
        if 0 in D:
            zero_ind = np.argwhere(D==0) 
            W = np.delete(W, zero_ind[:,0], axis=0)
            U = np.delete(U, zero_ind[:,0], axis=0)
            D = np.delete(D, zero_ind[:,0], axis=0)
            X = np.delete(X, zero_ind[:,0], axis=0)
            print('Updating centers, center and data overlap, len={}'.format(len(zero_ind)))
                
        denom = (U.T) @ W # [C,N] * [N,1] --> [C,1]
        W = W.repeat(X.shape[-1], axis=1) # [N,1] --> [N,R]
        numer = (U.T) @ (W * X) # [C,N] * [N,R] --> [C,R]

        new_center = numer / denom
        return new_center
    
    def next_u(self, X):
        alpha = 1/self.T1
        temp = cdist(X, self.centers)**2  # [400,3] --> [400,1,3] -->[400,3,3]
        
        if 0 in temp:
           zero_ind_exists = True
           zero_ind = np.argwhere(temp == 0)
           print('Updating U, center and data overlap, len={}'.format(len(zero_ind)))
           for ind in zero_ind:
               temp[ind[0], ind[1]] = 1
        else:
            zero_ind_exists = False
           
        denominator = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator = temp[:, :, np.newaxis] / denominator # [400,3,1]/[400,3,3] --> [400,3,3]
        denominator = denominator ** alpha
        res = 1/denominator.sum(2)
        if zero_ind_exists:
           for ind in zero_ind:
               res[ind[0], ...] = 0
               res[ind[0],ind[1]] = 1
        return res
    
    ## Trick from FCM for stable computation: put numer to denom
    def next_w(self, X):
        # based on updated u_{ij} and w_{i}
        # xi: observation
        # yj: current center
        # i : index for observation，total N
        # j : index for center，total C
        # U : [N,C] # numerator:[N,C] denominator:[N,1] --> update u_{ij} (公式1)
        # W : [N,1] # numerator:[N,1] denominator:[1,1] --> update w_{i}  (公式2)
        # X:  [N,R] # R: feature dimension
        alpha = -1 / self.T1
        beta = self.T1 / self.T2
        temp = cdist(X, self.centers)**2 
        
        if 0 in temp:
           zero_ind_exists = True
           zero_ind = np.argwhere(temp == 0)
           print('Updating W, center and data overlap, len={}'.format(len(zero_ind)))
           for ind in zero_ind:
               temp[ind[0], ind[1]] = 1
        else:
            zero_ind_exists = False

        temp = temp ** alpha
        temp = np.sum(temp, axis=1) # [N,C]
        denom = temp.reshape((X.shape[0], -1)).repeat(temp.shape[0], axis=1)
        res = (temp / denom) ** beta
        res = 1 / np.sum(res, axis=0)

        if zero_ind_exists:
           for ind in zero_ind:
               res[ind[0], ...] = 0
        return res[:,np.newaxis]

    # predict: original data
    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        U = self.next_u(X) 
        return np.argmax(U, axis=-1)
    
    # predict: new data
    def predict_new(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        return self.next_u(X) 

    def get_U(self):
        return self.U

    def get_W(self):
        return self.W

    def get_gif(self, X, iter, plot_boundary=False):
        fig, ax = plt.subplots(1,1, figsize=(8,8))
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
        
        ax.plot(self.centers[:,0],  self.centers[:,1], "*" , markersize = 10, c = 'blue') 

        for ind in range(self.n_clusters):
            ax.scatter(X[pred==ind,0], X[pred==ind,1], s=5, alpha=0.5, c=colors_tmp[ind])
            if plot_boundary:
               ax.scatter(mesh_grid_pred[cls_ind_list[ind],0],mesh_grid_pred[cls_ind_list[ind],1], c = colors_tmp[ind], s=point_size, alpha=alpha)

        ax.set_title('iter={}, T1={:.2f}, T2={:.2f}'.format(iter, self.T1, self.T2))
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
    model = ITISC_AO(args)
    model.fit(data)
    centers = model.centers
    pred = model.predict(data)
    print('centers\n', centers)
    
    if not args.gif: 
        plt.scatter(data[:,0],data[:,1], c=pred, s=10)
        plt.scatter(centers[:,0],centers[:,1], c='k', s=30)
        plt.show()      



