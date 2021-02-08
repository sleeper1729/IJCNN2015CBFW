import numpy as np
from PIL import Image
import scipy.signal
import cv2
import glob
import sparse_coding as sc
from sporco import signal
from config import *
import get_feature as gf
import time

mean = np.load("mean.npy")
var = np.linalg.inv(np.load("var.npy"))
dictionary = np.load("dictionary.npy")

def Mahalanobis_dist(vector):
    return np.sqrt(((vector-mean).transpose()).dot(var.dot(vector-mean)))

def detect(image):
    image = np.asarray(Image.open(image).convert("L"))
    image = np.asarray(Image.fromarray(image).resize((row,col))).astype(np.float64).reshape((row,col,1))

    start = time.time()

    low, high = signal.tikhonov_filter(image, lmbda=lmbda_tf)

    code, reconstructed=sc.reconstruct_and_sparse_coding(dictionary,high)

    hcf = gf.high_component_feature(high,code,reconstructed)
    hcf = np.array([[[hcf[0][i][j],hcf[1][i][j],hcf[2][i][j]] for j in range(row)] for i in range(col)])

    feature = [[list(hcf[x,y,:])+[gf.low_component_feature(low)[0][x][y],gf.low_component_feature(low)[1][x][y]] for y in range(col)] for x in range(row)]
    feature = np.array(feature)

    dist = np.array([[Mahalanobis_dist(feature[i][j]) for j in range(col)] for i in range(row)])

    point = np.where(dist>threshold)
    marked = image.reshape((row,col))
    for i in range(len(point[0])):
        marked[point[1][i]][point[0][i]]=0

    cv2.imwrite("../results/marked{0}.jpg".format(num),marked)

    print("elapsed time: {0}".format(time.time()-start))


if __name__=="__main__":
    num = 0
    import sys
    image = sys.argv[1]
    detect(image)


