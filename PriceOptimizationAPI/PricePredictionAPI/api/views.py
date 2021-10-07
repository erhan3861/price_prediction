import numpy as np
import pandas as pd
from .apps import ApiConfig
#from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

# our home page view
def home(request):    
    return render(request, 'index.html')


# our result page view
def result(request):
    price_class = int(request.GET['price_class'])
    shipping = int(request.GET['shipping'])
    rating_point = int(request.GET['rating_point'])
    rating_number = int(request.GET['rating_number'])
    seller_point = int(request.GET['seller_point'])
    

    result = getPrediction(shipping,rating_point,rating_number,seller_point,price_class)

    return render(request, 'result.html', {'result':result})



# custom method for generating predictions
def getPrediction(shipping,rating_point,rating_number,seller_point,price_class):
    import pickle
    model = pickle.load(open("price_prediction_model.sav", "rb"))
    scaled = pickle.load(open("scaler.sav", "rb"))
    prediction = model.predict(sc.transform([[shipping,rating_point,rating_number,seller_point,price_class]]))
    
    return prediction


     
import pandas as pd    
# import the data saved as a csv
df = pd.read_csv("D:/PriceOptimizationAPI/PricePredictionAPI/api/price_dynamics2.csv")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_shipping = le.fit_transform(df['shipping'])
le_seller_name = le.fit_transform(df['seller_name'])

df['shipping'] = le_shipping
df['seller_name'] = le_seller_name

print(df.head())
print(df.info())   
 
correlations=df.corr()
       
import seaborn as sns
from matplotlib import pyplot as plt
# Distribution plot on price


point = []
rating_number = []
discount_ratio = []
seller_point = []


print(df.head())

# veri kümesi
#shipping,rating_point,rating_number,seller_point,price_class
X = df.iloc[:, [5,6,7,9,10]].values
y = df.iloc[:,2].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)


import numpy as np
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=1000,random_state=42)
regressor.fit(X_train,y_train)


# saving model as a pickle
import pickle
pickle.dump(regressor,open("price_prediction_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))

    
"""
    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"
"""
"""
class PricePrediction2(APIView):
    def post(self, request):
        data = request.data
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        
            
        X_test = pd.Series(values).to_numpy().reshape(1, -1)
        X_test = sc.transform(X_test) 
        
        randomForest_reg_model = ApiConfig.model
        y_pred = randomForest_reg_model.predict(X_test)
        y_pred = pd.Series(y_pred)
        response_dict = {"Predicted Price = ": y_pred[0]}
        return Response(response_dict, status=200)
"""