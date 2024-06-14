from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import pickle
import os
import joblib
import sklearn
# Create your views here.
class Prediction(APIView):
    def make_prediction(self,arr):
        prediction_model=joblib.load(os.path.join(os.path.dirname(__file__), 'model', 'diabetes.pkl'))
        prediction=prediction_model.predict([arr])
        return prediction 

    def post(self,request,format=None):
        data=request.data.get('data',[])
        result=self.make_prediction(data)
        # data=[ 0.3429808,   1.41167241,  0.14964075, -0.09637905,  0.82661621, -0.78595734, 0.34768723,  1.51108316]
        return Response({"message":"The person is diabetic" if result==1 else "The person is not diabetic"})