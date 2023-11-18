### hyperparameter for ITISC
import argparse
parser = argparse.ArgumentParser(description='ITISC algorithms')
parser.add_argument('--T1',default = 1.0, type=float, help='parameter T1')
parser.add_argument('--T2',default = 0.5, type=float, help='parameter T2')
parser.add_argument('--m',default = 2.0, type=float, help='parameter m for FCM algorithm')

parser.add_argument('--N1',default = 200, type=int, help='# of points in cluster 1')
parser.add_argument('--N2',default = 200, type=int, help='# of points in cluster 2')
parser.add_argument('--N3',default = 200, type=int, help='# of points in cluster 3')
parser.add_argument('--N4',default = 200, type=int, help='# of points in cluster 4')
parser.add_argument('--N5',default = 200, type=int, help='# of points in cluster 5')
parser.add_argument('--N6',default = 200, type=int, help='# of points in cluster 6')
parser.add_argument('--N',default = 50, type=int, help='# uniform random noise')

parser.add_argument('--max_iter',default = 150, type=int, help='maximum number of iterations')
parser.add_argument('--random_state',default = 0, type=int, help='random seed') # data generation
parser.add_argument('--itisc_random_state',default = 0, type=int, help='random seed for itisc initialization')   # itisc random seed
parser.add_argument('--fcm_random_state',default = 0, type=int, help='random seed for fcm initialization')       # fcm random state
parser.add_argument('--kmeans_random_state',default = 0, type=int, help='random seed for kmeans initialization') # kmeans random state
parser.add_argument('--error',default = 1e-5, type=float, help='convergence error')

### ITISC initialization
parser.add_argument('--init_method', default='random', type=str, 
                    help='method to initialize clustering centers(default:random)',
                    choices=['kmeans','fcm', 'random','given','ITISC-R'])

# argment for generating synthetic 2D Gaussian generation
parser.add_argument('--C', default = 3, type=int, help='modeled cluster numbers') # required=True
parser.add_argument('--mean1', nargs='+', default=[1,0], type=float, help='true mean value for the 1-th cluster')
parser.add_argument('--mean2', nargs='+', default=[2,0], type=float, help='true mean value for the 2-th cluster')
parser.add_argument('--mean3', nargs='+', default=[3,0], type=float, help='true mean value for the 3-th cluster')
parser.add_argument('--mean4', nargs='+', default=[4,0], type=float, help='true mean value for the 4-th cluster')
parser.add_argument('--mean5', nargs='+', default=[5,0], type=float, help='true mean value for the 5-th cluster')
parser.add_argument('--mean6', nargs='+', default=[6,0], type=float, help='true mean value for the 6-th cluster')
parser.add_argument('--cov1', nargs='+', default=[0.2,0,0,0.2], type=float, help='1-th covariance matrix')
parser.add_argument('--cov2', nargs='+', default=[0.2,0,0,0.2], type=float, help='2-th covariance matrix')
parser.add_argument('--cov3', nargs='+', default=[0.2,0,0,0.2], type=float, help='3-th covariance matrix')
parser.add_argument('--cov4', nargs='+', default=[0.2,0,0,0.2], type=float, help='4-th covariance matrix')
parser.add_argument('--cov5', nargs='+', default=[0.2,0,0,0.2], type=float, help='5-th covariance matrix')
parser.add_argument('--cov6', nargs='+', default=[0.2,0,0,0.2], type=float, help='6-th covariance matrix')
parser.add_argument('--R', default = 1.0, type=float, help='Gaussian synthetic dataset generation hyperparameter')
# M-boundaryDist
parser.add_argument('--max_num', default = 10, type=int, help='Number of marginal points')
# analysis 
parser.add_argument('--model', default='itisc', type=str, help='clustering method', choices=['kmeans','fcm', 'itisc','dafcm','isfcm'])
parser.add_argument('--save_plot',action='store_true',help='whether to save plots')
# plot
parser.add_argument('--prob',default = 1/3, type=float, help='probability for isoprobability curve')
parser.add_argument('--plot',action='store_true',help='test code, plot')
parser.add_argument('--plot_boundary',action='store_true',help='whether to isoprobability curve')
parser.add_argument('--gif',action='store_true',help='whether to draw gif in ITISC')
parser.add_argument('--gif_gap', default = 1, type=int, help='record interval between gifs')
### hierarchical clustering 
parser.add_argument('--linkage', default='ward', type=str, help='linkage type for hierarchical clustering')

# python arg.py -l 1234 2345 3456 4567
# check python or jupyter
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

if is_notebook():
   args = parser.parse_args(args=[]) # jupyter
else:
   args = parser.parse_args() # python
