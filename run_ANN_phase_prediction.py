import pandas as pd
import numpy as np 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #stops display of tensorflow logs

import tensorflow as tf;
from tensorflow.keras import layers;
from tensorflow.keras import Model;
from tensorflow.keras import optimizers;

from sklearn.utils import shuffle


# Import local functions from 'f_ANN.py' and 'f_extract_input_data.py'
from f_ANN import f_ANN_model;
from f_extract_input_data import f_extract_input;


input_file_data = f_extract_input("Input_ANN.txt"); #dictionary that stores all data in Input txt file

x_feats = input_file_data['x']; # 'x' key value stored as x_feats (model features)
y_prop = input_file_data['y']; # 'y' key value stored as y_prop (phase_code)

print("\nReading database... ");
db_filename = input_file_data['database'][0];
db = pd.read_csv(db_filename, encoding='latin-1'); # database as pandas DataFrame
db = db[['alloy_name', 'phases'] + y_prop + x_feats]; #keeping only required data
print("Done.");

# dropping rows with NaN values in any column(Excluding alloy_name and phases columns)
db = db.dropna(axis='index', subset = y_prop + x_feats);
print('Dataset shape:',db.shape);

#creating directories for storing the results
print("\nCreating project directories ... ");
proj_dir = input_file_data['project_name'][0];
os.mkdir(proj_dir);

model_save_dir = input_file_data['project_name'][0]+'\\model_save\\';
os.mkdir(model_save_dir);

pred_save_dir = input_file_data['project_name'][0]+'\\prediction_save\\';
os.mkdir(pred_save_dir);
print("Done.");

""" Phase information is converted to binary vectors
    
    FCC: 1 - [1,0,0]
    BCC: 2 - [0,1,0]
    FCC + BCC: 3 - [1,1,0]
    FCC + Im: 4 - [1,0,1]
    BCC + Im: 5 - [0,1,1]
    FCC + BCC + Im: 6 - [1,1,1]
    Im: 7 - [0,0,1]
"""

phase_code = db['phase_code'];

print("\nConverting phase information to binary vector... ");
bin_code = []; #list to store binary vectors

for i in phase_code:

    if i==1:
        bin_code.append([1,0,0]);
    if i==2:
        bin_code.append([0,1,0]);
    if i==3:
        bin_code.append([1,1,0]);
    if i==4:
        bin_code.append([1,0,1]);
    if i==5:
        bin_code.append([0,1,1]);
    if i==6:
        bin_code.append([1,1,1]);
    if i==7:
        bin_code.append([0,0,1]);
        
print("Done.");
 
X = db[x_feats];
Y = bin_code;

X, Y = shuffle(X, Y, random_state=1);

phases = np.array(db['phases']).astype('str');
labels = db['alloy_name'] + ' (' + phases + ')';


# Dataframes to store loss and accuracy
loss_db = pd.DataFrame(data=np.arange(1,int(input_file_data['iterations'][0])+1,1) ,columns=['Iteration']);
acc_db = pd.DataFrame(data=np.arange(1,int(input_file_data['iterations'][0])+1,1) ,columns=['Iteration']);

K = 5; # value of K for K-fold cross-validation

L = len(Y);
K_L = int(L/K); #size (no. of rows) in each K validation set: here K_L = 40

X = np.array(X);
print('\nX-dataset shape:', X.shape);

Y = np.array(Y);
print('Y-dataset shape:', Y.shape);


# Running K-fold cross-validation
print("\nRunning %d-fold cross validation ... "%int(K));
for Kn in range(0,K):

    Kx_train = np.delete(X, np.s_[int(Kn*K_L):int(Kn*K_L+K_L)],0); #delete K_L no. of rows from trainx and store rest as K_trainx
    Kx_test = X[int(Kn*K_L):int(Kn*K_L+K_L)];

    Ky_train = np.delete(Y, np.s_[int(Kn*K_L):int(Kn*K_L+K_L)],0); #delete K_L no. of rows from trainx and store rest as K_trainx
    Ky_test = Y[int(Kn*K_L):int(Kn*K_L+K_L)];

    K_labels = labels[int(Kn*K_L):int(Kn*K_L+K_L)];
    x_axis = np.linspace(1,K_L,K_L);


    # Converting train and test set back to pandas dataframe to retain column titles
    Kx_train = pd.DataFrame(data=Kx_train,columns=x_feats);
    Kx_test = pd.DataFrame(data=Kx_test,columns=x_feats);


    # empty lists to store predictions
    pred_NN_train = [];
    pred_NN_test = [];

    # empty lists to store loss
    train_K_loss = [];
    val_K_loss = [];

    train_K_acc = [];
    val_K_acc = [];

    threshold = float(input_file_data['check_acc'][0]);
    check_acc = threshold-2;
    
    
    # Ensuring model coverges by putting condition on accuracy before model can move forward
    while check_acc < threshold:

        ANN_model = f_ANN_model(input_file_data);

        ANN_eval = ANN_model.fit(Kx_train,
                                 Ky_train,
                                 epochs = int(input_file_data['check_after_iterations'][0]),
                                 validation_data=(Kx_test, Ky_test),
                                 verbose=0);

        check_acc = ANN_eval.history['accuracy'][-1];

    train_K_loss += ANN_eval.history['loss'];
    val_K_loss += ANN_eval.history['val_loss'];

    train_K_acc += ANN_eval.history['accuracy'];
    val_K_acc += ANN_eval.history['val_accuracy'];

    n_stops = int((int(input_file_data['iterations'][0])-int(input_file_data['check_after_iterations'][0]))/int(input_file_data['save_after_iterations'][0]));

    
    for stop in range(1, n_stops+1):

        ANN_eval = ANN_model.fit(Kx_train,
                                 Ky_train,
                                 epochs = int(input_file_data['save_after_iterations'][0]),
                                 validation_data=(Kx_test, Ky_test),
                                 verbose=0);

        train_K_loss += ANN_eval.history['loss'];
        val_K_loss += ANN_eval.history['val_loss'];

        train_K_acc += ANN_eval.history['accuracy'];
        val_K_acc += ANN_eval.history['val_accuracy'];

        it_n = int(input_file_data['check_after_iterations'][0])+stop*int(input_file_data['save_after_iterations'][0]);
        print(str(input_file_data['project_name'][0])+': K'+str(Kn+1)+'-'+str(it_n)+' iterations');
    
        # Save model
        ANN_model.save(model_save_dir+proj_dir.split('.')[0]+'.stop'+str(it_n)+'_K'+str(Kn+1)+'_model.h5');
        
        # Reshape Kx_train and Kx_test to match format for ANN_model input
        Kx_train = np.array(Kx_train).reshape(Ky_train.shape[0],1,len(x_feats));
        Kx_test = np.array(Kx_test).reshape(Ky_test.shape[0],1,len(x_feats));

        pred_phases = pd.DataFrame(columns=['P_FCC','P_BCC','P_Im']);
        actual_phases = pd.DataFrame(columns=['A_FCC','A_BCC','A_Im']);
        
        # Make predictions for validation set
        count=0;
        for i in range(0,Ky_test.shape[0]):
            pred_phases.loc[count] = ANN_model.predict(Kx_test[i])[0];
            actual_phases.loc[count] = Ky_test[i];
            count+=1;

        # Save current K-set predictions
        K_labels_pd = pd.DataFrame(data=K_labels,columns=['alloy_name']);
        K_labels_pd = K_labels_pd.reset_index(drop=True);

        K_itr_result = pd.concat([K_labels_pd, actual_phases, pred_phases], axis=1);

        K_itr_result.to_csv(pred_save_dir+proj_dir.split('.')[0]+'.stop'+str(it_n)+'_K'+str(Kn+1)+'_pred.csv');

        # Reshape Kx_train and Kx_test back to initial shape so they can be fed back for training
        Kx_train = np.array(Kx_train).reshape(Ky_train.shape[0],len(x_feats));
        Kx_test = np.array(Kx_test).reshape(Ky_test.shape[0],len(x_feats));

    
    # Adding curent K-set loss/accuracy to main loss/accuracy database
    loss_db['K'+str(Kn+1)+'_train_loss'] = train_K_loss;
    loss_db['K'+str(Kn+1)+'_val_loss'] = val_K_loss;

    acc_db['K'+str(Kn+1)+'_train_acc'] = train_K_acc;
    acc_db['K'+str(Kn+1)+'_val_acc'] = val_K_acc;


# Save all loss/accuracy results to 'csv' file
loss_db.to_csv(proj_dir+'\\loss.csv');
acc_db.to_csv(proj_dir+'\\acc.csv');

print('Finished.');