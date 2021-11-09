import argparse
import logging
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

from numpy.random import seed

import tensorflow as tf
tf.random.set_seed(221)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

logging.getLogger().setLevel(logging.INFO)

def make_datasets_unbatched():
    data = pd.read_csv("heart.csv")
    data.head()
    
    data.apply(lambda x: sum(x.isnull()),axis=0)
    
    # List of variables with missing values
    
    vars_with_na=[var for var in data.columns if data[var].isnull().sum()>1]
    
    #Boolan variables
    bool_var=['sex','output','fbs', 'exng']
    #Categorical variables:cardinalty
    cat_var=['cp','restecg','slp', 'thall']
    #discrete variables
    num_var=['age', 'trtbps','chol', 'thalachh', 'oldpeak', 'caa']
    #remove outliers
    
    def removeOutlier(att, data):
        lowerbound = att.mean() - 3 * att.std()
        upperbound = att.mean() + 3 * att.std()
        #print('lowerbound: ',lowerbound,' -------- upperbound: ', upperbound )
        df1 = data[(att > lowerbound) & (att < upperbound)]
        #print((data.shape[0] - df1.shape[0]), ' number of outliers from ', data.shape[0] )
        #print(' ******************************************************')
        data = df1.copy()
        return data
    data = removeOutlier(data.trtbps, data)
    data = removeOutlier(data.chol, data)
    
    #resampling
    from sklearn.utils import resample
    
    # Separate Target Classes
    df_1 = data[data.output==1]
    df_2 = data[data.output==0]
    
    # Upsample minority class
    df_upsample_1 = resample(df_2, 
                                 replace=True,     # sample with replacement
                                 n_samples=163,    # to match majority class
                                 random_state=123) # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_1, df_upsample_1])
    
    # Display new class counts
    df_upsampled.output.value_counts()
    
    x = df_upsampled.drop('output', axis = 1)
    y = df_upsampled['output']
    
    #Split dataset

    
    x_train,x_test, y_train, y_test = tts(x,y, test_size = 0.2, random_state = 111)
    
    #Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train = train_dataset.cache().shuffle(2000).repeat()
    return train, test_dataset

def model(args):
    seed(1)
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=13))
    #model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    opt = args.optimizer
    model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    tf.keras.backend.set_value(model.optimizer.learning_rate, args.learning_rate)
    return model


def main(args):
    # MultiWorkerMirroredStrategy creates copies of all variables in the model's
    # layers on each device across all workers
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.AUTO)
    logging.debug(f"num_replicas_in_sync: {strategy.num_replicas_in_sync}")
    BATCH_SIZE_PER_REPLICA = args.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    # Datasets need to be created after instantiation of `MultiWorkerMirroredStrategy`
    train_dataset, test_dataset = make_datasets_unbatched()
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    # See: https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
    tf.data.experimental.AutoShardPolicy.DATA
    
    train_datasets_sharded  = train_dataset.with_options(options)
    test_dataset_sharded = test_dataset.with_options(options)
    
    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = model(args)
        # Keras' `model.fit()` trains the model with specified number of epochs and
        # number of steps per epoch. 
        multi_worker_model.fit(train_datasets_sharded,
                         epochs=50,
                         steps_per_epoch=30)

        eval_loss, eval_acc = multi_worker_model.evaluate(test_dataset_sharded, 
                                                    verbose=0, steps=10)
        # Log metrics for Katib
        logging.info("loss={:.4f}".format(eval_loss))
        logging.info("accuracy={:.4f}".format(eval_acc))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      metavar="N",
                      help="Batch size for training (default: 128)")
    parser.add_argument("--learning_rate", 
                      type=float,  
                      default=0.1,
                      metavar="N",
                      help='Initial learning rate')
    parser.add_argument("--optimizer", 
                      type=str, 
                      default='adam',
                      metavar="N",
                      help='optimizer')
    parsed_args, _ = parser.parse_known_args()
    main(parsed_args)
