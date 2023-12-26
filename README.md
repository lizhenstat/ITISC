# ITISC
This project provides python implementation of Information Theoretical Importance Sampling Clustering and Its Relationship with Fuzzy C-Means described in the following paper:
 * [Information Theoretical Importance Sampling Clustering and Its Relationship with Fuzzy C-Means. ](https://ieeexplore.ieee.org/document/10372109).

# Dependencies
Python 3.8.5, MATLAB R2022a, Ubuntu 16.04
* The Fuzzy C-Means algorithm is based on [the following repository](https://github.com/omadson/fuzzy-c-means/tree/master). 

# Code on synthetic dataset
The example is implemented in each python file.
ITISC Alternative Optimization algorithm. 
```bash
python ITISC_AO.py       
python ITISC_AO.py --gif # gif result is saved in gif subfolder
```
ITISC Reformulation algorithm(MATLAB installed required), the intermediate MATLAB result is saved in matlabCenters subfolder
```bash
python ITISC_R.py
python ITISC_R.py --gif # gif result is saved in gif subfolder
```
# Citing this work 
Please do not hesitate to leave us an issue if there is a problem. 
```latex
@article{zhang2023information,
  title={Information Theoretical Importance Sampling Clustering and Its Relationship with Fuzzy C-Means},
  author={Zhang, Jiangshe and Ji, Lizhen and Wang, Meng},
  journal={IEEE Transactions on Fuzzy Systems},
  year={2023},
  publisher={IEEE}
}
```


