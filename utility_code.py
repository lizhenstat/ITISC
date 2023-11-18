### utility function for ITISC
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import csv
import numpy as np
import random
from math import pi, cos, sin
from scipy.spatial.distance import cdist # l2: default
from math import pi, cos, sin
from hyperparam import args

def get_color_list():
    c1 = np.array([1,86,153])/255
    c2 = np.array([250,192,15])/255
    c3 = np.array([243,118,74])/255
    c4 = np.array([95,198,201])/255
    c5 = np.array([79,89,100])/255
    c6 = np.array([176,85,42])/255
    color_list = [c1,c2,c3,c4,c5,c6]
    return color_list

def get_color_tmp():
    colors_tmp = ['g', 'r', 'b', 'c', 'm', 'k','y','cyan','pink','gray']
    return colors_tmp

def save_obj(obj, name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, dir):
    save_path = os.path.join(dir, name + '.pkl')
    with open(save_path, 'rb') as f:
        return pickle.load(f)

def get_kmeans(args, data):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=args.C, random_state=args.kmeans_random_state).fit(data) # kmeans++ initialization
    kmeans_centers = kmeans.cluster_centers_
    kmeans_pred = kmeans.predict(data) # data:[N,R]
    print('Kmeans-centers\n', kmeans_centers)
    return kmeans, kmeans_centers, kmeans_pred

def get_fcm(args,data):
    from fcm_github import FCM
    fcm = FCM(args)
    fcm.fit(data)
    fcm_centers = fcm.centers
    fcm_pred = fcm.predict(data)
    print('fcm_centers\n', fcm_centers)
    return fcm, fcm_centers, fcm_pred

def get_hc(args, data):
    from hierarchical_clustering import HC
    hc = HC(C=args.C, linkage=args.linkage)
    hc.fit(data)
    hc_centers = hc.centers
    hc_pred = hc.pred
    print('hc_centers\n', hc_centers)
    return hc, hc_centers, hc_pred

### ITISC_AO
def get_itisc_ao(args, data):
    from ITISC_AO import ITISC_AO
    itisc = ITISC_AO(args)
    _ = itisc.fit(data)
    itisc_centers = itisc.centers
    itisc_pred = itisc.predict(data)
    print('itisc_ao_centers', itisc_centers)
    return itisc, itisc_centers, itisc_pred

### ITISC-R: matlab: log d(x,y)
def get_itisc_r(args, data):
    from ITISC_R import ITISC_R
    itisc = ITISC_R(args)
    _ = itisc.fit(data)
    itisc_centers = itisc.centers
    itisc_pred = itisc.predict(data)
    print('itisc_r_centers', itisc_centers)
    return itisc, itisc_centers, itisc_pred


### KL divergence between two Gaussian
def get_KL_two_multivariate_gaussian(mu_1, mu_2, sigma_1, sigma_2):
    n = sigma_1.shape[0]
    det_sigma_1 = np.linalg.det(sigma_1)
    det_sigma_2 = np.linalg.det(sigma_2)
    sigma_2_inv = np.linalg.inv(sigma_2) # inverse matrix of sigma_2
    result = np.log(det_sigma_2/det_sigma_1) - n + np.matrix.trace(sigma_2_inv @ sigma_1) + \
    np.transpose(mu_2 - mu_1) @ sigma_2_inv @ (mu_2 - mu_1)
    return 0.5 * result

### mesh grid for isoprobability curve
def get_meshgrid(data, h=0.025):
    # h: meshgrid gap 
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    data_pred = np.c_[xx.ravel(), yy.ravel()] # Obtain labels for each point in mesh. Use last trained model.
    return data_pred

def get_boundary_index(U_pred, cls_ind, prob, rtol=1e-2):
    # U_pred:  [N_new, C]
    # cls_ind: cluster center index
    # prob:    probability for isoprobability curve
    # rtol:    allowed error 1e-2
    cls_num = np.isclose(U_pred[:,cls_ind], prob, rtol)
    cls_num_ind  = np.argwhere(cls_num == 1)
    return cls_num_ind


def get_rotate_matrix(cov,rotate_degree):  
    cov = np.array(cov).reshape(2,2)
    theta = pi * rotate_degree
    rotate_matrix = [cos(theta), -sin(theta), sin(theta), cos(theta)]   # clockwise rotate
    # rotate_matrix = [cos(theta), sin(theta), -sin(theta), cos(theta)] # counter clockwise rotate
    rotate_matrix = np.array(rotate_matrix).reshape(2,2)
    tmp = rotate_matrix @ cov @ rotate_matrix.T
    tmp = tmp.flatten()
    return tmp

# extreme dataset: C=3
def get_extreme_3_clusters(args):
    args.mean1 = [1,0]
    args.mean2 = [8,0]
    args.mean3 = [4,8]
    args.cov1 = [0.8,0.4,0.4,0.8]
    args.cov2 = [0.8,0.4,0.4,0.8]
    args.cov3 = [0.8,-0.4,-0.4,0.8]
    args.N1 = 2
    args.N2 = 100
    args.N3 = 2
    return args 

# extreme dataset: C=2
def get_extreme_2_clusters(args):
    args.C = 2
    args.mean1 = [1,0]
    args.mean2 = [5,0]
    args.cov1 = [1,0.5,0.5,1]
    args.cov2 = [1,-0.5,-0.5,1]
    return args

    
def get_data(args):
    # args.R = 1.5
    N = 200
    args.N1 = N
    args.N2 = N
    args.N3 = N
    args.N4 = N
    args.N5 = N
    args.N6 = N
    ### C = 2
    if args.C == 2:
        args.mean1 = [args.R,0]
        args.mean2 = [-args.R,0]
        cov = [1,0,0,0.3]
        args.cov1 = get_rotate_matrix(cov,0.25)
        args.cov2 = get_rotate_matrix(cov,0.75)   
    ### C=3
    elif args.C == 3:
        args.mean1 = [args.R,0]
        args.mean2 = [-1 * args.R / np.sqrt(3),-1*args.R]
        args.mean3 = [-1 * args.R / np.sqrt(3),args.R]
        args.cov1 = [1,0,0,0.3]
        args.cov2 = get_rotate_matrix(args.cov1,1/3) # cov2 
        args.cov3 = get_rotate_matrix(args.cov1,2/3) # cov3
    ### C=4
    elif args.C == 4:
        args.mean1 = [args.R,args.R]
        args.mean2 = [args.R,-args.R]
        args.mean3 = [-args.R,-args.R]
        args.mean4 = [-args.R,args.R]
        cov = [1,0,0,0.1]
        rotate_degree = 0.5 
        args.cov1 = get_rotate_matrix(cov,0.25) # cov2
        args.cov2 = get_rotate_matrix(cov,0.75) # cov2
        args.cov3 = get_rotate_matrix(cov,1.25) # cov3
        args.cov4 = get_rotate_matrix(cov,1.75) # cov4
    ### C=6
    elif args.C == 6:
        tmp = np.sqrt(3)
        args.mean1 = [args.R/2,args.R/2*tmp] 
        args.mean2 = [-args.R/2,args.R/2*tmp] 
        args.mean3 = [-args.R,0] 
        args.mean4 = [-args.R/2,-args.R/2*tmp] 
        args.mean5 = [args.R/2,-args.R/2*tmp] 
        args.mean6 = [args.R,0]
        cov = [1,0,0,0.3]
        rotate_degree = 1/3 
        args.cov1 = get_rotate_matrix(cov,rotate_degree) # cov2
        args.cov2 = get_rotate_matrix(cov,rotate_degree*2) 
        args.cov3 = get_rotate_matrix(cov,rotate_degree*3) 
        args.cov4 = get_rotate_matrix(cov,rotate_degree*4) 
        args.cov5 = get_rotate_matrix(cov,rotate_degree*5) 
        args.cov6 = get_rotate_matrix(cov,rotate_degree*6)
    ### 
    print('mean1', args.mean1)
    print('mean2', args.mean2)
    print('mean3', args.mean3)
    print('mean4', args.mean4)
    print('mean5', args.mean5)
    print('mean6', args.mean6)

    print('cov1', args.cov1)
    print('cov2', args.cov2)
    print('cov3', args.cov3)
    print('cov4', args.cov4)
    print('cov5', args.cov5)
    print('cov6', args.cov6)
    return args


### 2D data generation
def generate_data(args):
    rs = np.random.RandomState(args.random_state)
    mean_list = []
    cov_list = []
    N_list = []

    for i in range(args.true_C):
        mean_list = mean_list + [eval('args.mean{}'.format(i+1))]
        cov_list = cov_list + [eval('args.cov{}'.format(i+1))]
        N_list = N_list + [eval('args.N{}'.format(i+1))]

    labels = []
    true_centers = []

    for i in range(args.true_C):
        mean_tmp = mean_list[i]
        cov_tmp = cov_list[i]
        N_tmp = N_list[i]

        cov_tmp = np.array(cov_tmp).reshape(2,2)
        x_tmp, y_tmp = rs.multivariate_normal(mean_tmp, cov_tmp, N_tmp).T
        if i == 0:   
           x = x_tmp 
           y = y_tmp
        else:
           x = np.concatenate((x, x_tmp), axis=0)
           y = np.concatenate((y, y_tmp), axis=0)

        labels = labels + [i] * N_tmp
        true_centers = true_centers + [mean_tmp]

    # random points
    labels = np.array(labels)
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    data = np.concatenate((x,y),1)
    return data, labels, np.array(true_centers)


def generate_data_ind(args):
    rs = np.random.RandomState(args.random_state)
    mean_list = []
    cov_list = []
    N_list = []
    for i in range(args.C):
        mean_list = mean_list + [eval('args.mean{}'.format(i+1))]
        cov_list = cov_list + [eval('args.cov{}'.format(i+1))]
        N_list = N_list + [eval('args.N{}'.format(i+1))]
    labels = []
    true_centers = []
    for i in range(args.C):
        mean_tmp = mean_list[i]
        cov_tmp = cov_list[i]
        N_tmp = N_list[i]
        cov_tmp = np.array(cov_tmp).reshape(2,2)
        x_tmp, y_tmp = rs.multivariate_normal(mean_tmp, cov_tmp, N_tmp).T
        if i == 0: 
           x = x_tmp 
           y = y_tmp
        else:
           x = np.concatenate((x, x_tmp), axis=0)
           y = np.concatenate((y, y_tmp), axis=0)
        labels = labels + [i] * N_tmp
        true_centers = true_centers + [mean_tmp]
    # random points
    labels = np.array(labels)
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    data = np.concatenate((x,y),1)
    ### 
    data_mean = np.mean(data, axis=0, keepdims=True)
    print('data_mean', data_mean) # data centroid
    dataMeanDist = cdist(data, data_mean)**2 # [N,1] 
    dataDistDesInd = dataMeanDist.squeeze().argsort()[::-1] # Data Distance Descending Index
    return data, labels, np.array(true_centers), data_mean, dataDistDesInd


