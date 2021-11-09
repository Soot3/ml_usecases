import argparse

def train(clean_data):
  #importing libraries
  import pandas as pd
  import joblib
  import numpy as np
  import tensorflow as tf
  from keras.layers import Input, Dropout
  from keras.layers.core import Dense 
  from keras.models import Model, Sequential, load_model
  from keras import regularizers

  data = joblib.load(clean_data)
  np.random.seed(10)
  tf.random.set_seed(10)
  X_train = data['X_train']
  act_func = 'elu'

  # Input layer:
  model=Sequential()
  # First hidden layer, connected to input vector X. 
  model.add(Dense(10,activation=act_func,
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(0.0),
                  input_shape=(X_train.shape[1],)
                )
          )

  model.add(Dense(2,activation=act_func,
                  kernel_initializer='glorot_uniform'))

  model.add(Dense(10,activation=act_func,
                  kernel_initializer='glorot_uniform'))

  model.add(Dense(X_train.shape[1],
                  kernel_initializer='glorot_uniform'))

  model.compile(loss='mse',optimizer='adam')

  # Train model for 100 epochs, batch size of 10: 
  NUM_EPOCHS=50
  BATCH_SIZE=10

  model.fit(np.array(X_train),np.array(X_train),
                    batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS,
                    validation_split=0.1,
                    verbose = 1)
  model.save('fit_model')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  train(args.clean_data)