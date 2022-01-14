#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 01:29:17 2018
@author: imlab
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn.model_selection as sk
from scipy.io import loadmat 
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import itertools
 
acc=[]
ep=[]
lss=[]
#epo=[1,2,3,4,5,6,7,8,9,10]

class_names = ['anomaly','normal']

#############
tf.compat.v1.reset_default_graph()
#############

###################

def plot_confusion_matrics(cm, classes, normalize=False, title='Confusion Matrics', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm=cm.astype('float')/cm.sum(axis-1)[:, np.newaxis]
		print('normalized cm')
	else:
		print('without normalization')
	print(cm)
	thresh=cm.max()/2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j]>thresh else 'black')
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('Full_dataset2c_ANomalyfull1.png')
	plt.show()
	
	
###################


#####################
#aa=np.load('UCF_features.npy', mmap_mode=None, allow_pickle=False, fix_imports=True)
#aa=np.load('Part1_UCF.mat', 'Part2_UCF.mat','Part3_UCF.mat','Part4_UCF.mat','Part5_UCF.mat','Part6_UCF.mat', mdict=None, appendmat=True)
#bb=loadmat('UCF_Train_full_label.mat', mdict=None, appendmat=True)
##YouTubeActions_TotalFeatures= UCF_features
##YouTubeActions_TotalFeatures=aa['TotalFeatures']
#
#b1=bb['DatabaseLabel'] ###new addition coux of my mistake
#
##DatabaseLabel=b1[:,[0,1]]
#DatabaseLabel=b1

#####################

#X_train, X_S, Y_train, Y_S = sk.train_test_split(UCF_features,DatabaseLabel,test_size=0.20,random_state = 42 ) #, shuffle=False

X_train, X_Validation, Y_train, Y_Validation = sk.train_test_split(UCF_Train_1_full_InceptionV3_anomaly_ful_videos,DatabaseLabel,test_size=0.20,random_state = 42 ) #, shuffle=False

#X_train, X_Validation, Y_train, Y_Validation = sk.train_test_split(X_train,Y_train,test_size=0.20,random_state = 42 ) #, shuffle=False
#X_train=X_train
#Y_train=Y_train
X_test=UCF_test_1_full_InceptionV3_anomaly_ful_videos
Y_test=DatabaseLabel000



hm_epochs = 20
n_classes = 2
batch_size = 256
# batch_size = 128
batch_size_val=128
chunk_size =1000
n_chunks =15
rnn_size = 512
 


trainSamples,FeaturesLength=Y_train.shape
ValidationSamples,FeaturesLength=Y_Validation.shape
loss=[];
Val_Accuracy=[];   

with tf.name_scope('Inputs'):
    x = tf.compat.v1.placeholder('float', [None, n_chunks,chunk_size],name="Features")
    y = tf.compat.v1.placeholder('float',name="Lables")

def recurrent_neural_network(x):
    
    
 #####################################################################

  
    W = {
            'hidden': tf.Variable(tf.random.normal([chunk_size, rnn_size])),
            'output': tf.Variable(tf.random.normal([rnn_size, n_classes]))
        }
    biases = {
            'hidden': tf.Variable(tf.random_normal([rnn_size], mean=1.0)),
            'output': tf.Variable(tf.random_normal([n_classes]))
        }


    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1,chunk_size])
    x = tf.nn.relu(tf.matmul(x, W['hidden']) + biases['hidden'])
    x = tf.split (x,n_chunks, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    Dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, output_keep_prob=0.5)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2,Dropout], state_is_tuple=True)
    # Get LSTM cell output
    outputs, final_states = tf.contrib.rnn.static_rnn(lstm_cells, x, dtype=tf.float32)
    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
#    lstm_last_output=tf.transpose(outputs, [1,0,2])
    # Linear activation
    
    return tf.matmul(outputs[-1], W['output']) + biases['output']
    
#####################################################################  







def train_recurrnet_neural_network(x):
    

    t = time.time()
    
    prediction= recurrent_neural_network(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    best_accuracy = 0.0
   
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2
                      (logits=prediction, labels=y) )
    optimizer = tf.compat.v1.train.AdamOptimizer(0.00000001).minimize(cost)
    with tf.compat.v1.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        tf.device('/gpu:0')
        sess.run(tf.compat.v1.global_variables_initializer())
        
       
#        print(sess.run(weights))
                          
        kk=0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            valdd=[]  
            k=0;
            for _ in range(int(trainSamples/batch_size)):
                epoch_x = X_train[k:k+batch_size,:]
                epoch_y = Y_train[k:k+batch_size,:]
                #print('epoch_x = ',epoch_x.size)
                #print('epoch_y = ',epoch_y.shape)
                
                epoch_x= epoch_x.reshape((batch_size, n_chunks, chunk_size ))
                
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                k=k+batch_size
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            loss.append(epoch_loss)
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            
            kk=0
            for _ in range(int(ValidationSamples/batch_size_val)):
                valdd.append(accuracy.eval({x:X_Validation[kk:kk+batch_size_val,:].reshape((-1,n_chunks, chunk_size)), y:Y_Validation[kk:kk+batch_size_val,:]}))
                kk = kk+batch_size_val
                if kk > ValidationSamples:
                    kk=0
                    
                    

            accuracy_out=np.mean(valdd)
            Val_Accuracy.append(accuracy_out)
            print('Validation Accuracy : ',accuracy_out,'  ||| Best Accuracy :',best_accuracy)
            #############################################
            
            ep.append(epoch)
            lss.append(epoch_loss)
            acc.append(accuracy_out)
            
            #############################################
            
            if  accuracy_out > best_accuracy:
                    best_accuracy=accuracy_out
                    saver = tf.train.Saver() 
                    save_path = saver.save(sess, "modelsave")
                    print("Model saved in file: %s" % save_path)
                    
                    
            
        PreLabels=sess.run(tf.argmax(prediction,1), feed_dict={x: X_Validation.reshape((-1,n_chunks, chunk_size))})
        Labels = Y_Validation.argmax(axis=1)
        confusion = tf.confusion_matrix(Labels, PreLabels).eval()
        elapsed = time.time() - t
        print('elapsed Time : ', elapsed) 
        
        plot_confusion_matrics(confusion, class_names, title='Confusion Matrics')  ###opps 
        
        return PreLabels, Labels, confusion
        
        #############################################
       
        
        
        
        
        ############################################
                    
         



        
        #Save the variables to disk.
#        save_path = saver.save(sess, "D:\\Speech Project\\Dataset\\BerlinImages\\BerlinImages\\1_Singleimages\\RNN Model For 257x45 double data spects\\model.ckpt")
#        print("Best Accuracy ==  " ,best_accuracy)
       # merged = tf.summary.merge_all()
       # writer=tf.summary.FileWriter("C:\\Users\\AMIN\\Anaconda2\\envs\\py35\\Lib\\site-packages\\tensorflow\\tensorboard\\otherLogs",sess.graph)
        

PreLabels, Labels, confusion = train_recurrnet_neural_network(x)


#sio.savemat('./YouTube model/PreLabels.mat', mdict={'PreLabels': PreLabels})
#sio.savemat('./YouTube model/Labels.mat', mdict={'Labels': Labels})   
#sio.savemat('./YouTube model/confusion.mat', mdict={'confusion': confusion})

########################

     
fig, axes = plt.subplots(2, 1)
fig.suptitle('Training Metrics')
 
axes[0].plot(ep,lss)           
axes[0].set_ylabel("Loss",fontsize=14)
axes[0].set_xlabel("Epoch",fontsize=14)
axes[0].grid(True)

axes[1].plot(ep,acc)            
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].grid(True)

fig.tight_layout()
fig.savefig('10_epochs_train15_2c.jpg')
plt.show()
        
########################

#fname="weights-waseem_code.hdf5"
#model.save_weights(fname,overwrite=True)
from sklearn.metrics import classification_report
print(classification_report(Labels,PreLabels, target_names=class_names))

