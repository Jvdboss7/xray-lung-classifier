from fastapi import FastApi,Request
from typing import Optional
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from xray.utils.main_utils import MainUtils

from xray.components.model_predictor import CarPricePredictor, CarData
from xray.constants import APP_HOST, APP_PORT
from xray.pipeline.pipeline import TrainPipeline
