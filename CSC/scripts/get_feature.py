import numpy as np
from PIL import Image
import scipy.signal
import cv2
import glob
import sparse_coding
from sporco import signal
from config import *

def low_component_feature(image_low):
    image_low = image_low.reshape((row,col))
    num_of_pixels = nbd[0]*nbd[1]
    local_mean = cv2.blur(image_low,nbd)/num_of_pixels
    
    local_var = cv2.blur((image_low-local_mean)**2,nbd)/num_of_pixels
    
    return local_mean, local_var

def high_component_feature(image_high,sparse_code,reconstructed):
    image_high = image_high.reshape((row,col))
    reconstructed = reconstructed.reshape((row,col))
    
    #平均値フィルター
    filt = np.ones(nbd)
    #再構成誤差
    reconstruction_error=scipy.signal.convolve2d((reconstructed-image_high)**2,filt,mode="same")
    
    #スパースコードのL1norm
    L1norm = np.sum([scipy.signal.convolve2d(np.abs(sparse_code[:,:,i]),filt,mode="same") for i in range(sparse_code.shape[-1])],axis=0)
    
    #L2norm
    L2norm = np.sum([np.sqrt(scipy.signal.convolve2d(sparse_code[:,:,i]**2,filt,mode="same")) for i in range(sparse_code.shape[-1])],axis=0)
    
    return reconstruction_error, L1norm, L2norm

def get_pixelwise_features(row,col,img_dir):
    imgs = glob.glob(img_dir+"/*")
    num_of_imgs = len(imgs)
    dictionary = np.load("dictionary.npy")
    
    hcfs = np.zeros((row,col,3,num_of_imgs)) #high_component_features
    lcfs = np.zeros((row,col,2,num_of_imgs)) #low_component_features
    
    #正常画像の再構成とスパースコーディング
    for i in range(num_of_imgs):
        img = np.asarray(Image.open(imgs[i]).convert('L').resize((row,col))).astype(np.float64).reshape((row,col))   
        #低周波成分と高周波成分への分離
        img_low, img_high = signal.tikhonov_filter(img, lmbda=lmbda_tf)
        sparse_code, reconstructed = sparse_coding.reconstruct_and_sparse_coding(dictionary,img_high)
        
        #低周波部分と高周波部分の特徴量を計算
        hcfs[:,:,0,i], hcfs[:,:,1,i], hcfs[:,:,2,i] = high_component_feature(img_high,sparse_code,reconstructed)
        lcfs[:,:,0,i], lcfs[:,:,1,i] = low_component_feature(img_low)
    
    #ピクセル毎の特徴量
    features=[[hcfs[i,j,0,k],hcfs[i,j,1,k],hcfs[i,j,2,k],lcfs[i,j,0,k],lcfs[i,j,1,k]] for i in range(row) for j in range(col) for k in range(num_of_imgs)]

    return features

def get_confidence_region(features):
    mean = np.mean(features,axis=0)
    features = features.transpose()
    var = np.cov(features)

    np.save("mean",mean)
    np.save("var",var)

    return mean, var

if __name__=="__main__":
    img_dir = "../image/train"
    features = get_pixelwise_features(row,col,img_dir)
    mean, var=get_confidence_region(np.array(features))