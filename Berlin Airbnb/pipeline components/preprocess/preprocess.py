import argparse
def preprocessing(data):
    import joblib
    import numpy as np
    import pandas as pd
    from math import sin, cos, sqrt, atan2, radians
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split


    df = joblib.load(data)
    
        # convert price to numeric value
    df["price"] = df["price"].apply(lambda x: x.replace("$", "")) # Remove dollar sign
    df["price"] = df["price"].apply(lambda x: x.replace(",", "")) # Remove thousand seperator
    df["price"] = df["price"].astype("float") # Cast the column into type float
    lowerbound = df["price"].mean() - 3 * df["price"].std()
    upperbound = df["price"].mean() + 3 * df["price"].std()
    df1 = df[(df["price"] > lowerbound) & (df["price"] < upperbound)]
    df = df1.copy()

    # create list for selected features
    selected = []

        # fill out the missing values in columns "host_is_superhost" and "host_identity_verified"
    df["host_is_superhost"] = df["host_is_superhost"].replace(np.NAN, "f")
    df["host_identity_verified"] = df["host_identity_verified"].replace(np.NAN, "f")
    # add the two columns to the 'selected' list
    selected.append('host_is_superhost')
    selected.append('host_identity_verified')


        # Calcuate the distance bwteen the listing and mianat tractions in Berlin
    # Formula to calculate distances
    def distance(lat1, lat2, lon1, lon2):
        R = 6373.0
        rlat1 = radians(lat1)
        rlat2 = radians(lat2)
        rlon1 = radians(lon1)
        rlon2 = radians(lon2)
        rdlon = rlon2 - rlon1
        rdlat = rlat2 - rlat1
        a = sin(rdlat / 2)**2 + cos(rlat1) * cos(rlat2) * sin(rdlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    # Top locations in Berlin
    toploc = {"hbf": [52.525293, 13.369359], 
              "txl": [52.558794, 13.288437], 
              "btor": [52.516497, 13.377683], 
              "museum": [52.517693, 13.402141], 
              "reichstag": [52.518770, 13.376166]}
    toploc = pd.DataFrame.from_dict(toploc)
    toploc_trans = toploc.transpose()
    toploc_trans.columns = ["latitude", "longitude"]

    # Construct distance columns
    dist = []
    for col in toploc.columns:
        df["dist_"+col] = df.apply(lambda x: distance(x.latitude, toploc[col][0], x.longitude, toploc[col][1]), axis=1)
        dist.append("dist_"+col)
        
    for col in dist:
        df[col+"_close"] = (df[col] < df[col].median())

    df["good_distance"] = df.apply(lambda x: any([x.dist_hbf_close, x.dist_txl_close, x.dist_museum_close, x.dist_reichstag_close]), axis=1)

    selected.append("good_distance")

    selected.append('room_type')

    # Amenities
    df["amenities"] = df["amenities"].apply(lambda x: x[1:-1].replace("\'", "").split(","))
    df["with_hair_dryer"] = df["amenities"].apply(lambda x: '"Hair dryer"' in x)
    df["lap_friendly"] = df["amenities"].apply(lambda x: '"Laptop friendly workspace"' in x)
    df["with_hanger"] = df["amenities"].apply(lambda x: "Hangers" in x)
    for i in ["with_hair_dryer", "lap_friendly", "with_hanger"]:
        selected.append(i)

    # minimum nights
    df["min_nights_greater_than_two"] = df["minimum_nights"] > 2
    selected.append("min_nights_greater_than_two")

        # Cleaning fee
    # Remove dollar sign
    df["cleaning_fee"][-df["cleaning_fee"].isna()] = df["cleaning_fee"][-df["cleaning_fee"].isna()].apply(lambda x: x.replace("$", "").replace(",", ""))
    df["cleaning_fee"] = df["cleaning_fee"].astype("float")
    df['cleaning_fee'].fillna(method="pad", inplace=True)
    selected.append('cleaning_fee')

    selected.append("accommodates")
    selected.append('cancellation_policy')
    selected.append("instant_bookable")

    # Convert string variables into categorical variables
    df["host_is_superhost"] = df["host_is_superhost"]=="t"
    df["host_identity_verified"] = df["host_identity_verified"]=="t"

    for col in df[selected].select_dtypes("bool").columns:
        df[col] = df[col].astype("int")
    
    data = df[selected]

    # Encode the categorical varibles  
    le = LabelEncoder()
      
    data['room_type']= le.fit_transform(data['room_type'])
    data['cancellation_policy']= le.fit_transform(data['cancellation_policy'])
    data['instant_bookable'] = np.where(data['instant_bookable'] == 't', 1, 0)

    #  Standardisation
    sc = StandardScaler()
    scaledFeatures = sc.fit_transform(data)

    # split to training and test set
    X = scaledFeatures
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": y_train, "Y_test": y_test}
    
    joblib.dump(data_dic, 'clean_data')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    preprocessing(args.data)
