from sklearn import cluster
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist # l2: default
import matplotlib.pyplot as plt

class HC:
    def __init__(self, C, linkage):
        self.n_clusters = C
        self.linkage = linkage
        self.model = None 
        self.centers = None
        self.pred = None
        self.clf = None

    def fit(self,X):
        self.model = cluster.AgglomerativeClustering(n_clusters = self.n_clusters, linkage=self.linkage)
        self.model = self.model.fit(X)
        self.pred = self.model.labels_ 
        self.clf = NearestCentroid() # hierarchical clustering centers 
        self.clf.fit(X, self.pred)
        self.centers = self.clf.centroids_ 
    
    # hierarchical clustering predict on new dataset
    def predict(self,X):
        pred_new = self.clf.predict(X)
        return pred_new

if __name__ == '__main__':
    from hyperparam import args
    from utility_code import get_data, generate_data_ind
    
    # data
    args.T2 = 0.1
    args.C = 3
    args.R = 1
    args = get_data(args)
    data, labels, true_centers, data_mean, dataDistDesInd = generate_data_ind(args)
    
    # model
    model = HC(C=args.C, linkage=args.linkage)
    model.fit(data)
    centers = model.centers
    pred = model.predict(data)
    print('centers\n', centers)
    
    plt.scatter(data[:,0],data[:,1], c=pred, s=10)
    plt.scatter(centers[:,0],centers[:,1], c='k', s=30)
    plt.show()  
