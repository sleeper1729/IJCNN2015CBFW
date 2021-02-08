from sporco import util, signal, cnvrep
from sporco.admm import cbpdn, ccmod
from sporco.dictlrn import dictlrn
import numpy as np
from PIL import Image
import scipy.signal
import cv2
from config import *

def reconstruct_and_sparse_coding(dictionary,image):
    opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 200,
                            'RelStopTol': 5e-3, 'AuxVarObj': False})
    sparse_coding = cbpdn.ConvBPDN(dictionary,image.astype(np.float64),lmbda=lmbda,opt=opt)
    
    #スパース表現を計算
    sparse_code = sparse_coding.solve()
    
    #画像再構成
    reconstructed = sparse_coding.reconstruct()
    
    #shapeをよしなに変換
    sparse_code = sparse_code.reshape((row,col,dictionary.shape[-1]))
    reconstructed = reconstructed.reshape((row,col))
    
    return sparse_code, reconstructed
