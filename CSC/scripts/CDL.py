
from sporco import util, signal, cnvrep
from sporco.admm import cbpdn, ccmod
from sporco.dictlrn import dictlrn
import numpy as np
from PIL import Image
import glob
from config import *


def learn_dictionary(row, col):
    train_img = glob.glob("../image/train/*")
    num_of_train_img = len(train_img)
    print("{0} training images".format(num_of_train_img))

    normals_low = np.zeros((row,col,num_of_train_img))
    normals_high = np.zeros((row,col,num_of_train_img))

    #画像の読み込み
    for i in range(num_of_train_img):
        normal = np.asarray(Image.open(train_img[i]).convert('L').resize((row,col))).astype(np.float64).reshape((row,col))
    
        #低周波成分と高周波成分への分離
        normals_low[:,:,i], normals_high[:,:,i] = signal.tikhonov_filter(normal, lmbda=lmbda_tf)


    #辞書の初期値
    D0 = np.random.randn(row_of_atom, col_of_atom, size_of_dict)
    #D1 = np.random.randn(16,16,16)

    #スパース表現を計算する時の設定
    #lmbda = 0.1
    optx = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1,
                'rho': 50.0*lmbda + 0.5, 'AutoRho': {'Period': 10,
                'AutoScaling': False, 'RsdlRatio': 10.0, 'Scaling': 2.0,
                'RsdlTarget': 1.0}})

    #辞書をアップデートするための設定
    #shape_of_dicts = (D0.shape,D1.shape)
    shape_of_dicts = D0.shape
    cri = cnvrep.CDU_ConvRepIndexing(shape_of_dicts, normals_high)
    optd = ccmod.ConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
                'rho': 10.0*cri.K, 'AutoRho': {'Period': 10, 'AutoScaling': False,
                'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}},
                method='ism')

    D0n = cnvrep.Pcn(D0, shape_of_dicts, cri.Nv, dimN=2, dimC=0, crp=True,
                     zm=optd['ZeroMean'])

    optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(D0n, cri.Cd, cri.M), cri.Nv),
             'U0': np.zeros(cri.shpD)})

    #スパース表現のアップデートオブジェクト
    xstep = cbpdn.ConvBPDN(D0n, normals_high, lmbda, optx)

    #辞書のアップデートオブジェクト
    #dstep = ccmod.ConvCnstrMODBase(None, normal_high, D0.shape,optd)
    dstep = ccmod.ConvCnstrMOD(None, normals_high, D0.shape, optd, method='ism')


    #アップデート
    opt = dictlrn.DictLearn.Options({'Verbose': True, 'MaxMainIter': MaxMainIter})
    d = dictlrn.DictLearn(xstep, dstep, opt)
    D0 = d.solve()
    print("DictLearn solve time: %.2fs" % d.timer.elapsed('solve'), "\n")
    np.save("dictionary",D0)

    return D0

if __name__=="__main__":
    learn_dictionary(row,col)



