from fastapi import FastAPI,Request, File
from typing import Optional
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from xray.utils.main_utils import MainUtils

#from xray.components.model_predictor import CarPricePredictor, CarData
from xray.constants import APP_HOST, APP_PORT
from xray.pipeline.train_pipeline import TrainPipeline
from xray.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")

#templates = Jinja2Templates(directory="templates")


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        await train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def prediction(image_file: bytes = File(description="A file read as bytes")):
    try:
        prediction_pipeline = PredictionPipeline()
        label = prediction_pipeline.run_pipeline(image_file)
        return JSONResponse(content= label, status_code=200)
    except Exception as e:
        return JSONResponse(content = f"Error Occurred! {e}", status_code=500)

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)