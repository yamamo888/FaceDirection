import os
os.environ["MKL_THREADING_LAYER"]="GNU"

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pickle,gzip
from collections import Counter
import DeepFried2 as df
from lbtoolbox.thutil import count_params
from lbtoolbox.augmentation import AugmentationPipeline,Cropper
from training_utils import dotrain,dostats,dopred
import pdb


## BiternionNet

def split(X,y,num,split=0.9):
    itr,ite,trs,tes = [],[],set(),set()
    for i, name in enumerate(num):
        # personIDを取り除く
        pID = int(name.split('_')[1])

        if pID in trs:
            itr.append(i)
        elif pID in tes:
            ite.append(i)
        else:
            if np.random.rand() < split:
                itr.append(i)
                trs.add(pID)
            else:
                ite.append(i)
                tes.add(pID)
    return (X[itr],y[itr],[num[i]for i in itr]),(X[ite],y[ite],[num[i]for i in ite])

############################################################
############################################################
## 定性的評価
def gaussfilter(num,sigma=0.3,retx=False,norm=np.sum):
    x = np.arange(-(num-1)/2,(num+1)/2)
    x /= np.max(x)
    y = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-x**2/(2*sigma**2))
    if norm is not None:
        y /= norm(y)
    return (x,y) if retx else y

def cyclic_filter(cycle,filter):
    cycle = np.pad(cycle,pad_width=len(filter)//2,mode='wrap')
    return np.correlate(cycle,filter,mode='valid')

def heatmap(preds,nbins=360):
    preds = (preds+3600)%360
    hm = np.zeros(nbins)

    mypreds = (preds/(360/nbins)).astype(int)
    cnt = Counter(mypreds)

    for ind,num in cnt.items():
        hm[ind] += num
    
    return hm

#def donut(ax.hm,**kw):
    #im = ax.imshow(dou)
############################################################
############################################################

class Flatten(df.Module):
    def symb_forward(self,symb_in):
        return symb_in.flatten(2)

    def mknet(self,*outlayers):
        return df.Sequential(
                #df.SpatialConvolutionCUDNN(3,24,(3,3)),
                df.SpatialConvolution(3,24,(3,3)),
                df.BatchNormalization(24),
                df.ReLU(),
                #df.SpatialConvolutionCUDNN(24,24,(3,3)),
                df.SpatialConvolution(24,24,(3,3)),
                df.BatchNormalization(24),
                #df.SpatialMaxPoolingCUDNN(2,2),
                df.SpatialMaxPooling((2,2)),
                #df.MaxPooling(2,2),
                df.ReLU(),
                #df.SpatialConvolutionCUDNN(24,48,(3,3)),
                df.SpatialConvolution(24,48,(3,3)),
                df.BatchNormalization(48),
                df.ReLU(),
                # df.PoolingCUDNN()?
                df.SpatialMaxPoolingCUDNN(48,48,3,3),
                #df.SpatialConvolution(48,48,(3,3)),
                df.BatchNormalization(48),
                df.SpatialMaxPooling((2,2)),
                df.ReLU(),
                #df.SpatialConvolutionCUDNN(48,64,(3,3)),
                df.SpatialConvolution(48,64,(3,3)),
                df.BatchNormalization(64),
                df.ReLU(),
                #df.SpatialConvolutionCUDNN(64,64,(3,3)),
                df.SpatialConvolution(64,64,(3,3)),
                df.BatchNormalization(64),
                df.ReLU(),
                df.Dropout(0.2),
                Flatten(),
                df.Linear(64*5*5,512),
                df.ReLU(),
                df.Dropout(0.5),
                *outlayers
        )

    def ensemble_degrees(self,angles):
        return np.arctan2(np.mean(np.sin(np.deg2rad(angles)),axis=0),np.mean(np.cos(np.deg2rad(angles)),axis=0))
    
    def dopred_deg(self,model,aug,x,batchSize=100):
        return np.rad2deg(dopred(model,aug,x,ensembling=self.ensemble_degrees,output2preds=lambda x: x,batchsize=batchSize))

    def maad_from_deg(self,preds,reals):
        return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(reals-preds)),np.cos(np.deg2rad(reals-preds)))))
    
    def ensemble_radians(self,angles):
        return np.arctan2(np.mean(np.sin(angles),axis=0),np.mean(np.cos(angles),axis=0))

    def dopred_rad(self,model,aug,x,batchSize=100):
        return dopred(model,aug,x,ensembling=ensemble_radians,output2preds=lambda x:x,batchSize=batchSize)
    
    def Error(self,preds,reals,epoch=-1):
        loss = self.maad_from_deg(preds,reals)
        mean = np.mean(loss,axis=1)
        std = np.std(loss,axis=1)
        print("loss: {:5.2f}° ±{:5.2f}° ".format(np.mean(mean),np.mean(std))) 
        print("std: {:5.2f}° ±{:5.2f}° ".format(np.std(mean),np.std(std))) 


class ModuloMADCriterion(df.Criterion):
    def symb_forward(self,symb_in,symb_tgt):
        self._assert_same_dim(symb_in,symb_tgt)
        return df.T.mean(np.abs(symb_in-symb_tgt)%360)

class VonMisesCriterion(df.Criterion):
    def __init__(self,kappa,radians=True):
        df.Criterion.__init__(self)
        self.kappa = kappa
        self.torad = 1 if radians else 0.0174532925
    
    def symb_forward(self,symb_in,symb_tgt):
        delta_rad = self.torad * (symb_in-symb_tgt)
        co = np.exp(2*self.kappa)
        return df.T.mean(co-df.T.exp(self.kappa*(1+df.T.cos(delta_rad))))

class Biternion(df.Module):
    def symb_forward(self,symb_in):
        return symb_in / df.T.sqrt((symb_in**2).sum(axis=1,keepdims=True))

    def deg2bit(angles_deg):
        angles_rad = np.deg2rad(angles_deg)
        return np.array([np.cos(angles_rad),np.sin(angles_rad)]).T
    def bit2deg(angles_bit):
        return (np.rad2deg(np.arctan2(angles_bit[:,1],angles_bit[:,0]))+360)% 360

class CosineCriterion(df.Criterion):
    def symb_forward(self,symb_in,symb_tgt):
        cos_angles = df.T.batched_dot(symb_in,symb_tgt)
        return df.T.mean(1-cos_angles)

if __name__ == "__main__":

    # X:[8694,3,50,50],y:[8694],num:[8694]
   
    data,label,Number = pickle.load(gzip.open('data/TownCentre.pkl.gz','rb'))
    # Train,TestData
    (trainX,trainY,trainNum),(testX,testY,testNum) = split(data,label,Number,split=0.9)
    #trainX,trainY = trainX.astype(df.floatX)/255,trainY.astype(df.floatX)
    trainX,trainY = trainX.astype(np.float32)/255,trainY.astype(np.float32)
    #testX,testY = testX.astype(df.floatX)/255,testY.astype(df.floatX)
    testX,testY = testX.astype(np.float32)/255,testY.astype(np.float32)
    pdb.set_trace() 
    ################### Pure Linear Regression #######################
    
    aug = AugmentationPipeline(trainX,trainY,Cropper((46,46)))
    
    #LinearTrain = [dotrain(net,df.MADCriterion(),aug,trainX,trainY[:,None]) for net in nets_shallow_linreg]
    nets_shallow_linreg = [df.Sequential(Flatten(),df.Linear(3*46*46,1,init=df.init.const(0))) for _ in range(10)]
    
    LinearPred = [Flatten().dopred_deg(net,aug,testX) for net in nets_shallow_linreg]
    Flatten().Error(LinearPred,testY[:,None])
    
    pdb.set_trace() 
    #################### Deep Linear Regression ##################
    
    nets_linreg = [Flatten().mknet(df.Linear(512,1,init=df.init.const(0)))for _ in range(5)]
    #DeepLinearTrain = 
    """ 
    for model in nets_linreg:
        dostats(model,aug,trainX,batchsize=1000)
    """
    DeepLinearPred = [Flatten().dopred_deg(net,aug,testX) for net in nets_linreg]
    Flatten().Error(DeepLinearPred,testY[:,None])
    pdb.set_trace() 

    ################### Deep Linear Regression in Radians ##############

    nets_linreg_rad = [Flatten().mknet(df.Linear(512,1,init=df.init.const(0))) for _ in range(5)]

    # DeepLinearRad = 

    for model in nets_linreg_rad:
        dostats(model,aug,trainX,batchSize=1000)

    DeepLinearRadPred = [Flatten().dopred_rad(net,aug,testX) for net in nets_linreg_rad]
    Flatten().Error(np.rad2deg(DeepLinearRadPred),testY[:,None])
    pdb.set_trace() 

    ################### Deep Linear Regression with Modulo ##############
    
    nets_linreg_mod = [Flatten().mknet(df.Linear(512,1,init=df.init.const(0))) for _ in range(5)]

    # DeepLinearMod = 

    for model in nets_linreg_mod:
        dostats(model,aug,trainX,batchSize=1000)

    DeepLinearModPred = [Flatten().dopred_deg(net,aug,testX) for net in nets_linreg_mod]
    Flatten().Error(DeepLinearModPred,testY[:,None])

    pdb.set_trace()
    
    ################# Von-Mises Criterion ##########################3
    nets_linreg_vm = [Flatten().mknet(df.Linear(512,1,init=df.init.const(0))) for _ in range(5)]

    # DeepLinearVM = 

    for model in nets_linreg_vm:
        dostats(model,aug,trainX,batchSize=1000)

    LinearVMPred = [Flatten().dopred_deg(net,aug,testX) for net in nets_linreg_vm]
    Flatten().Error(LinearVMPred,testY[:,None])
    pdb.set_trace() 

    ####################### Biternion ##############################
    nets_linreg_bt_cos = [Flatten().mknet(df.Linear(512,2,init=df.init.normal(0.01)),Biternion()) for _ in range(5)]

    # Biternion = 

    for model in nets_linreg_bt_cos:
        dostats(model,aug,trainX,batchSize=1000)

    BiternionPred = [Biternion().bit2deg(dopred_deg(net,aug,testX)) for net in nets_linreg_bt_cos]
    Flatten().Error(BiternionPred,testY[:,None])

    pdb.set_trace()
    """
    # plot
    fig,axes = plt.subplots(1,2,figsize=(8,4))
    trainY_hm = cyclic_filter(heatmap(trainY,nbins=3600),gaussfilter(41))
    testY_hm = cyclic_filter(heatmap(testY,nbins=3600),gaussfilter(41))
    
    dount(axes[0],trainY_hm/(len(trainY_hm)/400),bg=(201,201),R=50,aapow=40);
    dount(axes[1],testY_hm/(len(testY_hm)/400),bg=(201,201),R=50,aapow=40);
    """
