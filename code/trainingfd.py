
import os
import sys
os.environ["MKL_THREADING_LAYER"]="GNU"

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pickle,gzip
from collections import Counter
import pdb


class Data():

    def __init__(self):
        
        # class数指定
        self.classMode = int(sys.argv[1])
        # pickleName
        pName = 'fd_{}.pkl'.format(self.classMode)
        # Path 
        dataPath = './data'
        self.fullPath = os.path.join(dataPath,pName)
    
    # データの読み込み
    def split(self,X,y,ant,num,split=0.9):
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
        return (X[itr],y[itr],ant[itr],[num[i]for i in itr]),(X[ite],y[ite],ant[ite],[num[i]for i in ite])


class Training():

    def __init__(self):
        
        #### parameters ####
        
        
        self.nClass = int(sys.argv[1]) 
        self.hiddenCell = 124 # class
        self.outputCell = self.nClass # class
        #self.inputCell_reg = self.outputCell + self.inputCell # regression
        self.hiddenCell_reg = 124 # regression
        self.outputCell_reg = 1 # regression
        self.sd = 0.0 # min angle
        self.ed = 360.0 # max angle
        cr = (self.ed-self.sd)/self.nClass 
        self.sigma = round(cr,6) # class range
        self.batchCnt = 0
        self.batchCnt_test = 0

        ####################

    def conv2d_relu(self,inputs,w,b,stride):
        conv = tf.nn.conv2d(inputs,w,strides=stride,padding='SAME')+b
        conv = tf.nn.relu(conv)
        return conv
    
    def maxPooling(self,inputs):
        return tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    """
    def batchNormalization(self,inputs,phase_train=None,decay=0.99):
        eps = 1e-5
        Dout = inputs.get_shape()[-1]
        scale = tf.get_variable('scale',[Dout],init=tf.constant_initiializer([tf.ones([Dout])]))
        beta = tf.get_variable('beta',[Dout],init=tf.constant_initiializer([tf.zeros([Dout])]))
        pmean = tf.get_variable('pmean',[Dout],init=tf.constant_initiializer([tf.zeros([Dout])]),trainable=False)
        pvar = tf.get_variable('scale',[Dout],init=tf.constant_initiializer([tf.ones([Dout])]),trainable=False)
        
        if phase_train == None:
            return tf.nn.batch_normalization(inputs,pmean,pvar,beta,scale,eps)
        
        rank = len(inputs.get_shape())
        axes = range(rank-1)
        batch_mean,batch_var = tf.nn.moments(inputs,axes)

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def updata():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.nn.batch_normalizaiton(inputs,tf.identity(batch_mean),tf.identity(batch_var),beta,scale,eps)
        def average():
            train_mean = pmean.assign(ema.average(batch_mean))
            train_var = pvar.assign(ema.average(batch_var))
            with tf.control_dependencies([train_mean,train_var]):
                return tf.nn.batch_normalization(inputs,train_mean,train_var,beta,scale,eps)

        return tf.cond(phase_train,update,average)"""



    def weight_variable(self,name,shape):
        return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))
    
    def bias_variable(self,name,shape):
         return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))

    def softmax(self,inputs,w,b,keepProb):
         softmax = tf.matmul(inputs,w) + b
         softmax = tf.nn.dropout(softmax, keepProb)
         softmax = tf.nn.softmax(softmax)
         return softmax
    
    def fc_relu(self,inputs, w, b, keepProb):
         relu = tf.matmul(inputs,w) + b
         relu = tf.nn.dropout(relu, keepProb)
         relu = tf.nn.relu(relu)
         return relu

    def fc(self, inputs,w, b, keepProb):
         fc = tf.matmul(inputs,w) + b
         fc = tf.nn.dropout(fc, keepProb)
         return fc

    def nextBatch(self,x,y,lable,batchSize=100):
         batchRandInd = np.random.permutation(y.shape[0])

         sInd = batchSize * self.batchCnt
         eInd = sInd + batchSize

         batchX = x[batchRandInd[sInd:eInd]]
         batchY = y[batchRandInd[sInd:eInd]]
         batchLabel = label[batchRandInd[sInd:eInd]]

         if eInd + batchSize > y.shape[0]:
             self.batchCnt = 0
         else:
             self.batchCnt += 1

         return batchX,batchY,batchLabel
    
    def nextTestBatch(self,x,y,lable,batchSize=100):
         batchRandInd = np.random.permutation(y.shape[0])

         sInd = batchSize * self.batchCnt_test
         eInd = sInd + batchSize

         batchX = x[batchRandInd[sInd:eInd]]
         batchY = y[batchRandInd[sInd:eInd]]
         batchLabel = label[batchRandInd[sInd:eInd]]

         if eInd + batchSize > y.shape[0]:
             self.batchCnt_test = 0
         else:
             self.batchCnt_test += 1

         return batchX,batchY,batchLabel
         
        

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    def cnn(self,x,reuse=False):
        with tf.variable_scope('cnn') as scope:
            keepProb = 1.0
            if reuse:
                keepProb = 1.0
                scope.reuse_variables()
            
            # 1Layer
            convW1 = self.weight_variable('convW1',[3,3,3,24])
            convB1 = self.bias_variable('convB1',[24])
            conv1 = self.conv2d_relu(x,convW1,convB1,stride=[1,2,2,1])
            
            # 2Layer
            convW2= self.weight_variable('convW2',[3,3,24,24])
            convB2 = self.bias_variable('convB2',[24])
            conv2 = self.conv2d_relu(conv1,convW2,convB2,stride=[1,2,2,1])
            
            # 3Layer
            convW3 = self.weight_variable('convW3',[3,3,24,48])
            convB3 = self.bias_variable('convB3',[48])
            conv3 = self.conv2d_relu(conv2,convW3,convB3,stride=[1,2,2,1])
            
            # 4Layer
            convW4 = self.weight_variable('convW4',[3,3,48,48])
            convB4 = self.bias_variable('convB4',[48])
            conv4 = self.conv2d_relu(conv3,convW4,convB4,stride=[1,2,2,1])
            
            # 5Layer
            convW5 = self.weight_variable('convW5',[3,3,48,64])
            convB5 = self.bias_variable('convB5',[64])
            conv5 = self.conv2d_relu(conv4,convW5,convB5,stride=[1,2,2,1])
            
            # 6Layer
            convW6 = self.weight_variable('convW6',[3,3,64,64])
            convB6 = self.bias_variable('convB6',[64])
            conv6 = self.conv2d_relu(conv5,convW6,convB6,stride=[1,2,2,1])
            
            # convert to vector
            conv6 = tf.reshape(conv6,[-1,np.prod(conv6.get_shape().as_list()[1:])]) 
            
            return conv6


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    def classification(self,cnnX,reuse=False):
        with tf.variable_scope('classification') as scope:  
            keepProb = 0.7
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
            #input -> hidden
            w1 = self.weight_variable('w1',[np.prod(cnnX.get_shape().as_list()[1:]),self.hiddenCell])
            bias1 = self.bias_variable('bias1',[self.hiddenCell])
            
            h = self.fc_relu(cnnX,w1,bias1,keepProb) 
             
            #hidden -> output
            w2 = self.weight_variable('w2',[self.hiddenCell,self.outputCell])
            bias2 = self.bias_variable('bias2',[self.outputCell])
            
            output = self.fc(h,w2,bias2,keepProb)
           
            
            return output
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
            
    def regression(self,classX,reuse=False):
        
        with tf.variable_scope('regression') as scope:  
            keepProb = 0.7
            if reuse:
                keepProb = 1.0            
                scope.reuse_variables()
             
            w1_regression = self.weight_variable('w1_regression',[np.prod(classX.get_shape().as_list()[1:]),self.hiddenCell_reg])
            bias1_regression = self.bias_variable('bias1_regression',[self.hiddenCell_reg])
            
            h_regression = self.fc_relu(classX,w1_regression,bias1_regression,keepProb)
            
            w2_regression = self.weight_variable('w2_regression',[self.hiddenCell_reg,self.outputCell_reg])
            bias2_regression = self.bias_variable('bias2_regression',[self.outputCell_reg])
            
            
            y = self.fc(h_regression,w2_regression,bias2_regression,keepProb)
            
            return y

class Plot():
    def __init__(self):
        x = 12
    
    def LearningLoss(self,num,loss):
        plt.plot(num,loss)
        plt.savefig('Trainloss.png')
        #plt.savefig('Testloss.png')
    
    def Variance(self,gt,pred):
        plt.plot(gt,pred,'.',color='m',linestyle='None',label='MVR')
        plt.savefig('TrainVariance.png')
        #plt.savefig('TestVariance.png')

if __name__ == "__main__":
    

    myData = Data()
    
    ############# データの読み取り####################### 
    
    # 顔向きの クラス数
    nClass = myData.classMode
    # Path
    fullPath = myData.fullPath
    
    
    """
    #--------------------------------------------------
    #--------------------------------------------------
    #元データ
    # X:[8694,3,50,50],y:[8694],num:[8694]
    data,label,number = pickle.load(gzip.open('data/TownCentre.pkl.gz','rb'))
    # 正規化,float型に変更 
    data = data.astype(np.float32)/255,label.astype(np.float32)
    
    # Anotation
    sD = 0.0
    eD = 360.0
    iD = round((eD-sD)/nClass,6)
    Ds = np.arange(sD,eD,iD)
    
    flag = False
    for tmpY in label:
        
        oneHot = np.zeros(len(Ds))
        
        ind = 0
        for threD in Ds:
            if (tmpY>=threD) & (tmpY<threD+iD):
                oneHot[ind] = 1
            ind += 1
            
        if tmpY == 360.0:
            oneHot[-1] = 1
        
        tmpY = oneHot
        tmpY = tmpY[np.newaxis]
        

        if not flag:
            labelY = tmpY.astype(np.float32)
            flag = True
        else:
            labelY = np.concatenate((labelY,tmpY.astype(np.float32)),axis=0)
            print(tmpY)
            print(labelY)
    
    # Save
    with open(fullPath,'wb') as fp:
        pickle.dump(data,fp)
        pickle.dump(label,fp)
        pickle.dump(labelY,fp)
        pickle.dump(number,fp)

    #--------------------------------------------------
    #--------------------------------------------------
            
    """
    # Data load
    with open(fullPath,'rb') as fp:
        image = pickle.load(fp)
        angle = pickle.load(fp)
        label = pickle.load(fp)
        number = pickle.load(fp)
    
    # Train,TestData
    (trainX,trainY,trainLabel,trainNum),(teX,teY,teLabel,teNum) = myData.split(image[0],angle,label,number,split=0.9)
    teX,teY = teX.transpose(0,2,3,1),teY[:,np.newaxis]
    """
    # 顔向きの分布
    plt.hist(trainY,label="trainData")
    plt.hist(testY,label='testData')
    plt.savefig("FaceDistribution.png")
    """    
    ####################################################
    
    myTraining = Training()
    
    #################### parameters #####################
    
    nClass = myTraining.nClass # number of Class
    nReg = nClass # number of regression input
    lr = 1e-4 # traininglate
    sigma = myTraining.sigma # class length
    nNorm = 10 # Normalization range
    imgSize = 50 # size of images
    rgb = 3 # RGB
    batchSize = 1000
    sd = myTraining.sd # start angle
    nCent = sd + (sigma/2) # center angle

    ####################################################
    
    # placeholder
    x = tf.placeholder(tf.float32,shape=[None,imgSize,imgSize,rgb])
    y_ = tf.placeholder(tf.float32,shape=[None,1])
    y_label = tf.placeholder(tf.float32,shape=[None,nClass])
    
    ######################### CNN ##############################

    # CNN output
    cnn_op = myTraining.cnn(x)
    cnn_test_op = myTraining.cnn(x,reuse=True)
    
    ##################### Classification #######################
    
    # Classification output
    predict_class_op = myTraining.classification(cnn_op)
    predict_class_test_op = myTraining.classification(cnn_test_op,reuse=True)

    #--------------------------------------------------
    #--------------------------------------------------

    # loss(Classification) 
    loss_class = tf.losses.softmax_cross_entropy(y_label,predict_class_op)
    loss_class_test = tf.losses.softmax_cross_entropy(y_label,predict_class_test_op)

    #--------------------------------------------------
    #--------------------------------------------------
    
    # Optimizer(Classification)
    trainer_class = tf.train.AdamOptimizer(lr).minimize(loss_class)

    #--------------------------------------------------
    #--------------------------------------------------
    
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_label,1),tf.argmax(predict_class_op,1)),tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_label,1),tf.argmax(predict_class_test_op,1)),tf.float32))

    #--------------------------------------------------
    #--------------------------------------------------

    ####################################################
    
    
    # placeholder
    #x_r = tf.placeholder(tf.float32,shape=[None,reg_feature])
    #x_test_r = tf.placeholder(tf.float32,shape=[None,reg_feature_test]) 

    
    ################ Regression #######################
    # Maxclass
    maxClass  = tf.expand_dims(tf.cast(tf.argmax(predict_class_op,1),tf.float32),1)
    maxClass_test =  tf.expand_dims(tf.cast(tf.argmax(predict_class_test_op,1),tf.float32),1)
    
    # classCenter
    classCenter = maxClass*sigma+nCent
    classCenter_test = maxClass_test*sigma+nCent
    
    # redidual(True)
    residual_op = y_ - classCenter
    residual_test_op = y_ - classCenter_test
    
    # Magnify
    residual_normalization = (residual_op+(nNorm*sigma))/((nNorm*sigma)*2)
    residual_normalization_test = (residual_test_op+(nNorm*sigma))/((nNorm*sigma)*2)

    # x of the Refinined-Net 
    predict_center_class_cnn =  tf.concat((classCenter,cnn_op),axis=1)
    predict_center_class_cnn_test =  tf.concat((classCenter_test,cnn_test_op),axis=1)
    
    #--------------------------------------------------
    #--------------------------------------------------
    # Regression output   
    predict_regression_op= myTraining.regression(predict_center_class_cnn)
    predict_regression_test_op= myTraining.regression(predict_center_class_cnn_test,reuse=True)

    #--------------------------------------------------
    #--------------------------------------------------

    # exp(output)
    #predict_exp_regression_op = tf.exp(predict_regression_op) 
    predict_exp_regression_op = predict_regression_op
    #predict_exp_regression_test_op = predict_regression_test_op
    predict_exp_regression_test_op = predict_regression_test_op
    
    #--------------------------------------------------
    #--------------------------------------------------
    
    # loss(Regression)
    loss_reg = tf.reduce_mean((tf.abs(predict_exp_regression_op - residual_normalization)))        
    loss_reg_test = tf.reduce_mean((tf.abs(predict_exp_regression_test_op - residual_normalization_test)))        
    
    #--------------------------------------------------
    #--------------------------------------------------
    
    # Optimizer(Regression)
    regressionVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="regression") 
    trainer_regression = tf.train.AdamOptimizer(lr).minimize(loss_reg,var_list=regressionVars)
    
    
    #--------------------------------------------------
    #--------------------------------------------------
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80,allow_growth=True))
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    #--------------------------------------------------
    #--------------------------------------------------
    
    myPlot = Plot()

    #--------------------------------------------------
    #--------------------------------------------------
    for epoch in range(50):
        
        for itr in range(100000):
            
            batchX,batchY,batchLabel = myTraining.nextBatch(trainX,trainY,trainLabel,batchSize)
            
            # change shape [batchSize,RGB,Hight,Weight]->[batchSize,Hight,Weight,RGB]
            # vector -> 
            batchX,batchY = batchX.transpose(0,2,3,1),batchY[:,np.newaxis]
            #--------------------------------------------
            #--------------------------------------------
            # Classification
            _,ClassLossTrain,PredCenterTrain = sess.run([trainer_class,loss_class,classCenter],feed_dict={x:batchX,y_:batchY,y_label:batchLabel})
            
            # Normalization
            #ResidualNormalization = np.where(Residual<-(nNorm*sigma),-(nNorm*sigma),np.where(Residual>(nNorm*sigma),(nNorm*sigma),Residual))
            #b1_r3,b2_r3,b3_r3 = np.where(b1residual<-(nNorm*sigma),0,np.where(b1residual>(nNorm*sigma),1,b1residual)),np.where(b2residual<-(nNorm*sigma),0,np.where(b2residual>(nNorm*sigma),1,b2residual)),np.where(b3residual<-(nNorm*sigma),0,np.where(b3residual>(nNorm*sigma),1,b3residual))
            
            # Regression
            _,RegressionLossTrain,PredRTrain = sess.run([trainer_regression,loss_reg,predict_exp_regression_op],feed_dict={x:batchX,y_:batchY,y_label:batchLabel})
             
            PredRegNormTrain = PredRTrain*(nNorm*sigma*2)-(nNorm*sigma)
            PredRegTrain = PredCenterTrain+PredRegNormTrain
            #--------------------------------------------
            #--------------------------------------------
            if itr % 250 == 0:
                print("------------------------------------")
                print("iteration: %d"%itr)
                print("ClassTrainloss: %f RegTrainloss: %f"%(ClassLossTrain,RegressionLossTrain))
                
                ######## Evaluation (MAE & STD) ###########
                # Vanilla Classification
                CM_ = np.mean(np.abs(batchY-PredCenterTrain))
                #CV_ = np.var(batchY-PredCenterTrain)
                CS_ = np.std(batchY-PredCenterTrain)
                # Magnified
                MM_ = np.mean(np.abs(batchY-PredRegTrain))
                #MV_ = np.var(batchY-PredRegTrain)
                MS_ = np.std(batchY-PredRegTrain)
                print("------------------------------------")
                print("Ground:",batchY[:10])
                print("VC:",PredCenterTrain[:10])
                print("MVR:",PredRegTrain[:10])
                print("------------------------------------")
                print("VC Mean %f"%CM_)
                #print("VC Var %f"%CV_)
                print("VC Std. %f"%CS_)
                print("------------------------------------")
                print("MVR Mean %f"%MM_)
                #print("MVR Var %f"%MV_)
                print("MVR Std. %f"%MS_)
                
                ############################################
            #-----------------------------------------------
            #-----------------------------------------------
        
            if itr % 500 == 0:
                testX,testY,testLabel =  myTraining.nextTestBatch(teX,teY,teLabel)
                # Classification
                ClassLossTest,ResidualTest,PredCenterTest = sess.run([loss_class_test,residual_test_op,classCenter_test],feed_dict={x:testX,y_:testY,y_label:testLabel})
                    
                # Normalization
                #ResidualNormalizationTest = np.where(ResidualTest<-(nNorm*sigma),-(nNorm*sigma),np.where(ResidualTest>(nNorm*sigma),(nNorm*sigma),ResidualTest))
                #b1_r3,b2_r3,b3_r3 = np.where(b1residual<-(nNorm*sigma),0,np.where(b1residual>(nNorm*sigma),1,b1residual)),np.where(b2residual<-(nNorm*sigma),0,np.where(b2residual>(nNorm*sigma),1,b2residual)),np.where(b3residual<-(nNorm*sigma),0,np.where(b3residual>(nNorm*sigma),1,b3residual))
                
                # Regression
                RegressionLossTest,predR = sess.run([loss_reg_test,predict_exp_regression_test_op],feed_dict={x:testX,y_:testY,y_label:testLabel})

                
                PredRegNorm = predR*(nNorm*sigma*2)-(nNorm*sigma)
                PredReg = PredCenterTest+PredRegNorm
                
                print("------------------------------------")
                print("itr: %d"%itr)
                print("ClassTrainloss: %f RegTrainloss: %f"%(ClassLossTrain,RegressionLossTrain))
                print("ClassTestloss: %f RegTestloss: %f"%(ClassLossTest,RegressionLossTest))

                
                #--------------------------------------------
                #--------------------------------------------
                ######## Evaluation (MAE & STD) ############

                # Vanilla Classification
                CM = np.mean(np.abs(testY-PredCenterTest))
                #CV = np.var(testY-PredCenterTest)
                CS = np.std(testY-PredCenterTest)
                # Magnified
                MM = np.mean(np.abs(testY-PredReg))
                #MV = np.var(testY-PredReg)
                MS = np.std(testY-PredReg)
                
                print("------------------------------------")
                print("VC Mean %f"%CM)
                #print("VC Var %f"%CV)
                print("VC Std. %f"%CS)
                print("------------------------------------")
                print("MVR Mean %f"%MM)
                #print("MVR Var %f"%MV)
                print("MVR Std. %f"%MS)
                print("------------------------------------")
