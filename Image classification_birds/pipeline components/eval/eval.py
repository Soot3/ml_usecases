import argparse
def eval(img_folder, keras_metrics, model_json, km_model, pytorch_metrics, pytorch_model):
  import joblib
  import tensorflow as tf
  from tensorflow.keras.models import model_from_json
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  
  test_directory=f'{img_folder}/test'

  test_datagen=ImageDataGenerator(rescale=1/255)


  test_generator=test_datagen.flow_from_directory(test_directory,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                class_mode='sparse',batch_size=256)  
  keras_metrics=joblib.load(keras_metrics)
  json_file = joblib.load(model_json)
  pytorch_metrics=joblib.load(pytorch_metrics)
  loss = keras_metrics['loss']
  accuracy = keras_metrics['test']
  
  loaded_model_json = json_file
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(km_model)
  
  opt=tf.keras.optimizers.RMSprop(lr=0.0001)
  loaded_model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=opt)
  score = loaded_model.evaluate(test_generator,verbose=0)
  print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
  print("/n/n ***************** /n/n")
  print('pytorch_metrics /n')
  print(pytorch_metrics)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_folder')
  parser.add_argument('--keras_metrics')
  parser.add_argument('--model_json')
  parser.add_argument('--km_model')
  parser.add_argument('--pytorch_metrics')
  parser.add_argument('--pytorch_model')
  args = parser.parse_args()
  eval(args.img_folder,args.keras_metrics,args.model_json,args.km_model,args.pytorch_metrics,args.pytorch_model)