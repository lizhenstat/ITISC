### ITISC-R: MATLAB
import os 
import random
import imageio
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.linalg import norm
import matlab.engine # import after scipy
import matplotlib.pyplot as plt

class ITISC_R:
    def __init__(self, args, initCenters=None):
        self.T1 = args.T1
        self.T2 = args.T2
        assert self.T1 > 0
        assert self.T2 > 0
        self.C = args.C
        self.centers = None
        self.historyCenters = None
        self.itisc_random_state = args.itisc_random_state
        self.kmeans_random_state = args.kmeans_random_state
        self.fcm_random_state = args.fcm_random_state
        self.init_method = args.init_method
        self.initCenters = initCenters
        
        # 初始化
        self.itisc_random_state = args.itisc_random_state
        self.kmeans_random_state = args.kmeans_random_state
        self.args = args # FCM

        # 画图
        self.gif = args.gif
        self.plot_boundary = args.plot_boundary # isoprobability curve
        self.prob = args.prob
        

    def init_centers(self, X):
        if not self.initCenters is None:
           initCenters = self.initCenters
        elif self.init_method == 'random':
            random.seed(self.itisc_random_state)
            xmax = np.max(X, axis=0)
            xmin = np.min(X, axis=0)
            initCenters = []
            for c in range(self.C):
                tmp = [random.uniform(tmp_min, tmp_max) for tmp_min, tmp_max in zip(xmin, xmax)]
                initCenters.append(tmp)
            self.initCenters = np.array(initCenters)
        elif self.init_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.C, random_state=self.kmeans_random_state).fit(X)
            self.initCenters = kmeans.cluster_centers_ 
        elif self.init_method == 'fcm':
            from fcm_github import FCM
            fcm = FCM(self.args)
            fcm.fit(X)
            self.initCenters = fcm.centers
        return self.initCenters
            
    def fit(self, X):
        initCenters = self.init_centers(X)  
        matlabFileName = 'matlabCenters/MultiGaussian.mat'
        pythonFileName = 'matlabCenters/matlabResult.mat'
        matlabDir = os.path.join(os.getcwd(),'models')
        outputFile = 'matlabCenters/matlabResult'
        scipy.io.savemat(matlabFileName, dict(initCenters = initCenters, data = X, T1 = self.T1, T2 = self.T2, outputFile=outputFile)) # 将数据转化为matlab的格式
        eng = matlab.engine.start_matlab()
        eng.addpath(matlabDir, nargout=0) # add dir to the matlab search directory 
        eng.ITISCFunction(matlabFileName)  # Matlab quasi-newton
        mat = scipy.io.loadmat(pythonFileName) # load matlab data
        self.historyCenters = mat['historyCenters']
        # exitflag=0:other status，exitflag=1:converge
        self.exitflag = mat['exitflag'][0][0] # matlab exit flag
        self.centers = self.historyCenters[-1]
        self.U = self.next_u(X) # according to current centers
        self.W = self.next_w(X) # according to current centers 
        if self.gif:
           self.get_gif(X)
        return self.centers
      
    def next_u(self, X):
        # based on updated u_{ij} and w_{i}
        # xi: observation
        # yj: current center
        # i : index for observation，total N
        # j : index for center，total C
        # U : [N,C] # numerator:[N,C] denominator:[N,1] --> update u_{ij} (公式1)
        # W : [N,1] # numerator:[N,1] denominator:[1,1] --> update w_{i}  (公式2)
        # X:  [N,R] # R: feature dimension
        alpha = 1/self.T1
        temp = cdist(X, self.centers)**2  # [400,3] --> [400,1,3] -->[400,3,3]
        denominator = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator = temp[:, :, np.newaxis] / denominator # [400,3,1]/[400,3,3] --> [400,3,3]
        denominator = denominator ** alpha
        res = 1/denominator.sum(2)
        return res
    
    def next_w(self, X):
        alpha = -1 / self.T1
        beta = self.T1 / self.T2
        temp = cdist(X, self.centers)**2 
        temp = temp ** alpha
        temp = np.sum(temp, axis=1) 
        denom = temp.reshape((X.shape[0], -1)).repeat(temp.shape[0], axis=1)
        res = (temp / denom) ** beta 
        res = 1 / np.sum(res, axis=0)
        return res[:,np.newaxis]

    # predict original dataset
    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        U = self.next_u(X) 
        return np.argmax(U, axis=-1)

    # predict new dataset
    def predict_new(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        return self.next_u(X)

    def get_U(self):
        return self.U

    def get_W(self):
        return self.W

    def get_gif_iter(self, X, iter):
        def u_given_center(X, centers):
            if len(X.shape) == 1:
                X = np.expand_dims(X, axis=0)
            # same as next_u with given center
            alpha = 1/self.T1
            temp = cdist(X, centers)**2 
            denominator = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
            denominator = temp[:, :, np.newaxis] / denominator # [N,C,1]/[N,C,C] --> [N,C,C]
            denominator = denominator ** alpha
            res = 1/denominator.sum(2)
            return res
        
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        point_size = 3
        centersize = 50
        alpha = 0.8
        colors_tmp = ['g', 'r', 'b', 'c', 'm', 'k','y','cyan','pink','gray']
        color_tmp = colors_tmp[:self.C]
        
        x_U_pred = u_given_center(X, self.historyCenters[iter,...]) # data pred at current iteration
        pred = np.argmax(x_U_pred, axis=-1)
        
        if  self.plot_boundary:
            rtol = 1e-2
            mesh_grid_pred = get_meshgrid(X)
            mesh_grid_U_pred = u_given_center(mesh_grid_pred, self.historyCenters[iter,...])
            cls_ind_list = []
            for ind in range(self.C):
                cls_ind_tmp = get_boundary_index(mesh_grid_U_pred, ind, self.prob, rtol=rtol)
                cls_ind_list = cls_ind_list + [cls_ind_tmp]
                
        ax.scatter(self.historyCenters[iter,:,0],  self.historyCenters[iter,:,1], marker="*" , s = centersize, c = 'blue', label='ITISC-R')
        for ind in range(self.C):
            ax.scatter(X[pred==ind,0], X[pred==ind,1], s=5, alpha=0.5, c=colors_tmp[ind])
            if self.plot_boundary:
               ax.scatter(mesh_grid_pred[cls_ind_list[ind],0],mesh_grid_pred[cls_ind_list[ind],1], c = colors_tmp[ind], s=point_size, alpha=alpha)
               
        ax.set_title('iter={},T1={},T2={}'.format(iter,self.T1, self.T2))
        ax.legend()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image, fig
    
    def get_gif(self, X):
        image_list = []
        for iteration in range(self.historyCenters.shape[0]):
            image, fig= self.get_gif_iter(X, iteration)
            image_list = image_list + [image]
            fig.clf()
        imageio.mimsave('./gif/ITISC_R_T1={}_T2={}.gif'.format(self.T1, self.T2), image_list, fps=1)


if __name__ == '__main__':
    from hyperparam import args
    from utility_code import get_data, generate_data_ind, get_meshgrid, get_boundary_index
    
    # data
    args.T2 = 0.1
    args.C = 3
    args.R = 1
    args = get_data(args)
    data, labels, true_centers, data_mean, dataDistDesInd = generate_data_ind(args)
    
    # model
    model = ITISC_R(args)
    model.fit(data)
    centers = model.centers
    pred = model.predict(data)
    print('centers\n', centers)
    
    if not args.gif:
        plt.scatter(data[:,0],data[:,1], c=pred, s=10)
        plt.scatter(centers[:,0],centers[:,1], c='k', s=30)
        plt.show()  




