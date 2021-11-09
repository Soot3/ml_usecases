import argparse
import logging
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications import ResNet101V2

logging.getLogger().setLevel(logging.INFO)




def make_datasets_unbatched():
    BUFFER_SIZE = 10000
    
    
    
    train_directory='input/train'
    val_directory='input/valid'
    test_directory='input/test'

    train_datagen=ImageDataGenerator(rescale=1/255)
    val_datagen=ImageDataGenerator(rescale=1/255)
    test_datagen=ImageDataGenerator(rescale=1/255)

    train_dataset=train_datagen.flow_from_directory(train_directory,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 class_mode='sparse',batch_size=256)

    val_dataset=val_datagen.flow_from_directory(val_directory,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 class_mode='sparse',batch_size=256)

    test_dataset=test_datagen.flow_from_directory(test_directory,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 class_mode='sparse',batch_size=256)
    return train_dataset, val_dataset, test_dataset


convlayer=ResNet101V2(input_shape=(224,224,3),weights='imagenet',include_top=False)
convlayer.trainable = False

def model(args):
    model=Sequential()
    model.add(convlayer)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(260,activation='softmax'))
    
    opt = args.optimizer
    
    model.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
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
    train_dataset, val_dataset, test_dataset= make_datasets_unbatched()
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    val_dataset = val_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    # See: https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA

    train_datasets_sharded  = train_dataset.with_options(options)
    val_datasets_sharded  = val_dataset.with_options(options)
    test_dataset_sharded = test_dataset.with_options(options)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = model(args)

    # Keras' `model.fit()` trains the model with specified number of epochs and
    # number of steps per epoch. 
    multi_worker_model.fit(train_datasets_sharded,validation_data=val_datasets_sharded,
                         epochs=5,
                         steps_per_epoch=10)

  
    eval_loss, eval_acc = multi_worker_model.evaluate(test_dataset_sharded, 
                                                    verbose=0, steps=10)


    # Log metrics for Katib
    logging.info("loss={:.4f}".format(eval_loss))
    logging.info("accuracy={:.4f}".format(eval_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                      type=int,
                      default=128,
                      metavar="N",
                      help="Batch size for training (default: 128)")
    parser.add_argument("--learning_rate", 
                      type=float,  
                      default=0.001,
                      metavar="N",
                      help='Initial learning rate')
    parser.add_argument("--optimizer", 
                      type=str, 
                      default='adam',
                      metavar="N",
                      help='optimizer')

    parsed_args, _ = parser.parse_known_args()
    main(parsed_args)
