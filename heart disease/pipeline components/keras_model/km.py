import argparse
def km(clean_data):
  import joblib
  from sklearn import metrics
  import pandas as pd
  from numpy.random import seed
  seed(1)
  import tensorflow as tf
  tf.random.set_seed(221)
  from tensorflow import keras
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
  data = joblib.load(clean_data)
  x_train = data['X_train']
  y_train = data['Y_train']
  x_test = data['X_test']
  y_test = data['Y_test']

  keras_model = Sequential()
  keras_model.add(Dense(100, activation='relu', input_dim=13))
  keras_model.add(BatchNormalization())
  keras_model.add(Dense(40, activation='relu'))
  keras_model.add(Dropout(0.2))
  keras_model.add(Dense(1, activation='sigmoid'))

  keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  keras_model.fit(x_train, y_train, epochs=100)

  y_pred=keras_model.predict_classes(x_test)
  
  eval_results = keras_model.evaluate(x_test,y_test, verbose=0) 
  loss = round(eval_results[0],4)
  accuracy = round((eval_results[1]), 2)
  print("\nLoss, accuracy on test data: ")
  print("%0.4f %0.2f%%" % (eval_results[0], \
    eval_results[1]*100))
  model_json = keras_model.to_json()
  weights = keras_model.save_weights('km.h5')
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  keras_metrics = {'loss':loss, 'test':accuracy, 'report':df_classification_report, 'model': [model_json, weights]}
  joblib.dump(keras_metrics,'keras_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  km(args.clean_data)
