import numpy as np
#from sklearn import preprocessing
from scipy import io
#np.random.seed(2021)


#%% system parameters
# antennas
M = 8
# reflecting elements
N = 96
# user number
K = 4
# noise variance
sigma_2 = 1
# Pmax
Pmax = 100
#calculate the SNR in dB
SNR = Pmax / sigma_2


#%% training
# training parameters
total_num = 10

# load from dataset
#dataset = io.loadmat(r'.\baselines\迭代优化\WSR-maximization-for-RIS-system-master\fig4\bl_data_%d_%d_%d_10_samples.mat'%(M,N,K))
dataset = io.loadmat('./bl_data_%d_%d_%d_10_samples.mat'%(M,N,K))
G_0 = dataset['G_list'][:,:total_num*M]
G_0 = np.reshape(G_0,(N,total_num,M))  
G_0 = np.transpose(G_0,(1,0,2)) 

print(G_0.shape) # (1,N,M)

# load from dataset
hr_0 = dataset['Hr_list'][:,:total_num*N]
hr_0 = np.reshape(hr_0,(K,total_num,N))
hr_0 = np.transpose(hr_0,(1,0,2))

print(hr_0.shape) # (1,K,N)

G_and_hr_0 = np.concatenate([np.reshape(G_0,(total_num,-1)),np.reshape(hr_0,(total_num,-1))],axis=-1)

# ((N*M+K*N),2)
G_and_hr_0 = np.expand_dims(G_and_hr_0,-1)
G_and_hr_0 = np.concatenate([np.real(G_and_hr_0),np.imag(G_and_hr_0)],axis=-1)

#data = np.expand_dims(G_and_hr_0,0)
data = G_and_hr_0
print(data.shape)

# fake label, not gonna be used
label = np.zeros((total_num,1))
print(label.shape)


W_list = dataset['W_list']
if K==1:
    W_list = np.expand_dims(W_list,-1)    
W_list = np.expand_dims(W_list,-1)
W_list = np.concatenate([np.real(W_list),np.imag(W_list)],axis=-1)
theta_list = np.angle(dataset['theta_list'])


#%% define the model
import tensorflow as tf
seed = np.random.randint(2021)
tf.set_random_seed(seed)

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    

from tensorflow.keras.layers import Activation,Multiply, GlobalAveragePooling1D, Add, Dense, Conv1D, Flatten, Reshape, Input, BatchNormalization, Concatenate, Lambda
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD


class W_Layer(tf.keras.layers.Layer):
  def __init__(self, output_shape):
    super(W_Layer, self).__init__()
#    self.output_shape = output_shape
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",shape=[M,K,2])
  def call(self, input):
    W_real = tf.expand_dims(self.kernel[:,:,0],0)
    W_imag = tf.expand_dims(self.kernel[:,:,1],0)
    energy = 0
    for i in range(K):
        w_k_real = W_real[:,:,i] #(M,1)
        w_k_imag = W_imag[:,:,i]
        energy = energy + tf.reduce_sum(w_k_real**2) + tf.reduce_sum(w_k_imag**2)    

    W_real = W_real/tf.sqrt(energy)*np.sqrt(Pmax)
    W_imag = W_imag/tf.sqrt(energy)*np.sqrt(Pmax)
    
    gamma_list = []
    
    for i in range(K):
        H_k = input[:,i] 
        H_k_real = H_k[:,:,:,0] #(?=1,1,8)
        H_k_imag = H_k[:,:,:,1]


        w_k_real = W_real[:,:,i:i+1] #(8,1) 
        w_k_imag = W_imag[:,:,i:i+1]
        
        RR = tf.matmul(H_k_real,w_k_real)
        RI = tf.matmul(H_k_real,w_k_imag)
        IR = tf.matmul(H_k_imag,w_k_real)
        II = tf.matmul(H_k_imag,w_k_imag)
        H_k_w_k_real = RR-II #(?=1,1,1)
        H_k_w_k_imag = RI+IR
        energy_signal = tf.reduce_sum(H_k_w_k_real**2,keepdims=True) + tf.reduce_sum(H_k_w_k_imag**2,keepdims=True) 
            
        
        energy_interference = 0
        for j in range(K):
            if j!=i:
                w_k_real = W_real[:,:,j:j+1] #(8,1) 
                w_k_imag = W_imag[:,:,j:j+1]
        
                RR = tf.matmul(H_k_real,w_k_real)
                RI = tf.matmul(H_k_real,w_k_imag)
                IR = tf.matmul(H_k_imag,w_k_real)
                II = tf.matmul(H_k_imag,w_k_imag)
                H_k_w_k_real = RR-II #(?=1,1,1)
                H_k_w_k_imag = RI+IR
                energy_interference = energy_interference + \
                        tf.reduce_sum(H_k_w_k_real**2) + tf.reduce_sum(H_k_w_k_imag**2)   
                        
        gamma_k = energy_signal/(energy_interference+sigma_2)
        gamma_list.append(gamma_k)
    
    gamma_list = tf.stack(gamma_list,axis=-1) # (?=1,K)
    gamma_list = gamma_list[:,0,0,:]
    
    return gamma_list


class H_Layer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(H_Layer, self).__init__()
    self.num_outputs = num_outputs
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",shape=[self.num_outputs])
  def call(self, input):
    G = input[:,:M*N]  
    G = tf.reshape(G,(-1,N,M,2))
    hr = input[:,M*N:]
    hr = tf.reshape(hr,(-1,K,N,2))      
     
    G_real = G[:,:,:,0]
    G_imag = G[:,:,:,1]
    
    hr_real = hr[:,:,:,0]
    hr_imag = hr[:,:,:,1]
    
    theta_real = tf.expand_dims(tf.diag(tf.cos(self.kernel)),0)
    theta_imag = tf.expand_dims(tf.diag(tf.sin(self.kernel)),0)
    
    RR = tf.matmul(theta_real,G_real)
    RI = tf.matmul(theta_real,G_imag)
    IR = tf.matmul(theta_imag,G_real)
    II = tf.matmul(theta_imag,G_imag)
    theta_G_real = RR-II
    theta_G_imag = RI+IR
   
    H_list = []
    
    for i in range(K):
        hr_k_real = hr_real[:,i:i+1]
        hr_k_imag = hr_imag[:,i:i+1]
    
        RR = tf.matmul(hr_k_real,theta_G_real)
        RI = tf.matmul(hr_k_real,theta_G_imag)
        IR = tf.matmul(hr_k_imag,theta_G_real)
        II = tf.matmul(hr_k_imag,theta_G_imag)
        
        H_k_real = RR-II
        H_k_imag = RI+IR
        
        H_k_real = tf.expand_dims(H_k_real,-1)
        H_k_imag = tf.expand_dims(H_k_imag,-1)

        H_k = tf.concat([H_k_real,H_k_imag],axis=-1)
        
        H_list.append(H_k)
    
    H_list = tf.stack(H_list,axis=1) # (?=1,K,1,M,2)
    
    return H_list


def bf_nn(M,N,K,init_lr): 
    def minus_sum_rate(y_true,y_pred):
        loss = -tf.log(1+y_pred)/np.log(2)
        loss = tf.reduce_sum(loss)
        return loss

    G_and_hr = Input(shape=(M*N+K*N,2)) 
    H = H_Layer(N)(G_and_hr) #(?=1,K,1,M,2)
    gamma = W_Layer([M,K])(H)
    
    model = Model(inputs=G_and_hr, outputs=gamma)
    model.compile(loss=minus_sum_rate, optimizer=Adam(lr=init_lr))
    model.summary()

    return model

max_epochs = 10000
batch_size = 1
best_model_path = './models/%d_%d_%d.h5'%(M,N,K)


#%%    
performance_list = []
epochs_list = []
init_lr = 1e-1
model = bf_nn(M,N,K,init_lr)

import time
t = 0

for i in range(total_num):
#for i in [0,1]: # certain samples
    print('Data sample %d'%i)  
    # initialization
    model.layers[1].set_weights([theta_list[i]])
#    model.layers[2].set_weights([W_list[i]])
    model.layers[2].set_weights([np.random.randn(M,K,2)])
#    checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
#    reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.1,patience=10,verbose=1,mode='auto',min_delta=1e-6,min_lr=1e-3)
    early_stopping = EarlyStopping(monitor='loss',min_delta=1e-6,patience=25)
    t_start = time.time()
    # 记录loss_history可能涉及数据在CPU和GPU之间的交互，从而增大GPU训练时间
    loss_history = model.fit(data[i:i+1],label[i:i+1],epochs=max_epochs,batch_size=batch_size,verbose=0,\
                                 callbacks=[early_stopping])
    t_end = time.time()
    print(t_end - t_start)
    if i>0: # ignore the time of the first sample due to GPU warm up
        t = t + t_end - t_start
    losses = loss_history.history['loss']
    performance_list.append(np.min(losses)) # best loss
    epochs_list.append(len(losses))
    
print(-np.array(performance_list))
print(-np.mean(performance_list))
print(np.mean(epochs_list))

print(t/(total_num-1)*1000)

from matplotlib import pyplot as plt
plt.plot(-np.array(losses))

#io.savemat('./results/new/init_2_adam_1e-1.mat',{'loss':-np.array(losses)})


