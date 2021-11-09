import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import preprocessing
  from sklearn.model_selection import train_test_split

  import re
  from imblearn.over_sampling import SMOTE
  df = joblib.load(data)

  df =df.dropna(axis = 0 , how = 'any')

  def age(dob):
      yr = int(dob[-2:])
      if yr >=0 and yr < 20:
          return yr + 2000
      else:
          return yr + 1900

  df['Date.of.Birth'] = df['Date.of.Birth'].apply(age)
  df['DisbursalDate'] = df['DisbursalDate'].apply(age)

  df['Age'] = df['DisbursalDate'] - df['Date.of.Birth']
  df = df.drop( ['DisbursalDate', 'Date.of.Birth'], axis=1)

  df['AVERAGE.ACCT.AGE_yrs'] = df['AVERAGE.ACCT.AGE'].apply(lambda x: re.search(r'\d+(?=yrs)', x).group(0)).astype(np.int)
  df['AVERAGE.ACCT.AGE_mon'] = df['AVERAGE.ACCT.AGE'].apply(lambda x: re.search(r'\d+(?=mon)', x).group(0)).astype(np.int)
  df = df.drop('AVERAGE.ACCT.AGE', axis=1)

  df['CREDIT.HISTORY.LENGTH_yrs'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x: re.search(r'\d+(?=yrs)', x).group(0)).astype(np.int)
  df['CREDIT.HISTORY.LENGTH_mon'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x: re.search(r'\d+(?=mon)', x).group(0)).astype(np.int)
  df = df.drop('CREDIT.HISTORY.LENGTH', axis=1)
  pri_columns = ['PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS',
            'PRI.ACTIVE.ACCTS','SEC.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS','SEC.OVERDUE.ACCTS',
            'PRI.CURRENT.BALANCE','SEC.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT','SEC.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT','SEC.DISBURSED.AMOUNT',
            'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT']

    #Creating and Sorting Columns

  df['NO_OF_ACCTS'] = df['PRI.NO.OF.ACCTS'] + df['SEC.NO.OF.ACCTS']

  df['ACTIVE_ACCTS'] = df['PRI.ACTIVE.ACCTS'] + df['SEC.ACTIVE.ACCTS']

  df['OVERDUE_ACCTS'] = df['PRI.OVERDUE.ACCTS'] + df['SEC.OVERDUE.ACCTS']

  df['CURRENT_BALANCE'] = df['PRI.CURRENT.BALANCE'] + df['SEC.CURRENT.BALANCE']

  df['SANCTIONED_AMOUNT'] = df['PRI.SANCTIONED.AMOUNT'] + df['SEC.SANCTIONED.AMOUNT']

  df['Total_AMOUNT'] = df['PRI.DISBURSED.AMOUNT'] + df['SEC.DISBURSED.AMOUNT']

  df['INSTAL_AMT'] = df['PRIMARY.INSTAL.AMT'] + df['SEC.SANCTIONED.AMOUNT']

  df['INACTIVE_ACCTS'] = df['NO_OF_ACCTS'] - df['ACTIVE_ACCTS']

  df.drop(pri_columns, axis=1, inplace=True)

 
  columns_unique = ['UniqueID','MobileNo_Avl_Flag',
          'Current_pincode_ID','Employee_code_ID',
          'NO.OF_INQUIRIES','State_ID',
          'branch_id','manufacturer_id','supplier_id', 'Driving_flag',	'Passport_flag']
  df = df.drop(columns=columns_unique)

  objects = df.select_dtypes('object').columns.tolist()
  le = preprocessing.LabelEncoder()
  df[objects] = df[objects].apply(le.fit_transform) 

  X = df.drop(['loan_default'], axis=1)
  y = df['loan_default']

  scaler = preprocessing.RobustScaler()
  X = scaler.fit_transform(X)


  # Split the data into training and testing sets 
  x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = .3, random_state = 33)


  sm = SMOTE(random_state=2)
  x_train, y_train = sm.fit_resample(x_train, y_train.ravel())
  
  data = {"X_train": x_train,"X_test": x_test, "Y_train": y_train,"Y_test": y_test}
  joblib.dump(data,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)