import os
import joblib
from django.apps import AppConfig
from django.conf import settings
import warnings


class ApiConfig(AppConfig):
    name = 'api'
    MODEL_FILE = os.path.join(settings.MODELS, "PriceRandomForestRegressionModel.joblib")
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      model = joblib.load(MODEL_FILE)