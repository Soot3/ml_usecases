import argparse
def km(img_folder):
  import joblib
  from numpy.random import seed
  import numpy as np
  seed(1)
  import tensorflow as tf
  tf.random.set_seed(221)
  from tensorflow import keras
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,Activation
  from tensorflow.keras.applications import ResNet101V2
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  
  train_directory=f'{img_folder}/train'
  val_directory=f'{img_folder}/valid'
  test_directory=f'{img_folder}/test'
  train_datagen=ImageDataGenerator(rescale=1/255)
  val_datagen=ImageDataGenerator(rescale=1/255)
  test_datagen=ImageDataGenerator(rescale=1/255)
  train_generator=train_datagen.flow_from_directory(train_directory,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=256)

  val_generator=val_datagen.flow_from_directory(val_directory,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=256)

  test_generator=test_datagen.flow_from_directory(test_directory,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=256)


  convlayer=ResNet101V2(input_shape=(224,224,3),weights='imagenet',include_top=False)
  convlayer.trainable = False
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

  opt=tf.keras.optimizers.RMSprop(lr=0.0001)
  model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=opt)
  history=model.fit(train_generator,validation_data=val_generator,
          epochs=5)

  loss = history.history['val_loss']
  accuracy = history.history['val_accuracy']
  print("\nLoss, accuracy on test data: ")
  print(loss[-1], accuracy[-1])
  model_json = model.to_json()
  weights = model.save_weights('km.h5')

  keras_metrics = {'loss':loss[-1], 'test':accuracy[-1]}
  joblib.dump(model_json, 'model_json')
  joblib.dump(weights, 'km.keras')
  joblib.dump(keras_metrics,'keras_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_folder')
  args = parser.parse_args()
  km(args.img_folder)
