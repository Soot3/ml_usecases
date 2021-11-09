import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder, RobustScaler
  from sklearn.model_selection import train_test_split
  df = joblib.load(data)
  df.dropna(inplace=True)
  features1= ['ActiveSubs','DroppedBlockedCalls','HandsetModels','OutboundCalls','OffPeakCallsInOut','OverageMinutes','PeakCallsInOut','ReceivedCalls','RetentionOffersAccepted','CustomerID']
  df.drop(columns=features1, inplace=True)
  # Imputting the mean price instead of Unknown
  df['HandsetPrice'] = df['HandsetPrice'].replace('Unknown',82.24)
  df['HandsetPrice']=df['HandsetPrice'].astype(float)
  dum = pd.get_dummies(df[['CreditRating', 'Occupation', 'MaritalStatus']], drop_first=True)
  objects = df.select_dtypes('object').columns.tolist()
  le = LabelEncoder()
  df[objects] = df[objects].apply(le.fit_transform) 
  df = pd.concat([df,dum], axis = 1)
    # selecting features, X
  x = df.drop(columns=['Churn','NewCellphoneUser','HandsetRefurbished','ChildrenInHH','NotNewCellphoneUser','TruckOwner','OwnsComputer','HandsetWebCapable','RespondsToMailOffers','Homeownership','BuysViaMailOrder',
  'HasCreditCard','RVOwner','ReferralsMadeBySubscriber','NonUSTravel','RetentionCalls','AdjustmentsToCreditRating','MadeCallToRetentionTeam','OwnsMotorcycle','OptOutMailings','CallForwardingCalls', 'CreditRating', 'Occupation', 'MaritalStatus'])
    # selecting labels, y
  y = df['Churn']
  scaler = RobustScaler()
  X = scaler.fit_transform(x.astype(float))
  X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.3, random_state=9)
  data = {"train": {"X": X_train, "y": Y_train},"test": {"X": X_test, "y": Y_test, "features":X}}
  joblib.dump(data,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)
