import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

from sentiment.exception import SentimentException
from sentiment.pipeline.training_pipeline import TrainPipeline
from sentiment.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI()

@app.post("/train")
def training_route():
    try:
        training_pipeline = TrainPipeline()
        if training_pipeline.is_pipeline_running:
            return Response("Training Pipeline is Still running!!!!!")
        if not training_pipeline.run_pipeline()['status']:
            return JSONResponse(content= {"msg": training_pipeline.run_pipeline()["msg"]})
        training_pipeline.run_pipeline()
        return JSONResponse(content= {"msg": training_pipeline.run_pipeline()["msg"]})
    except Exception as e:
        raise SentimentException(e, sys)

@app.post("/predict", response_class=JSONResponse)
def prediction_route(text: str):
    try:
        prediction_pipeline = PredictionPipeline()
        pro = prediction_pipeline.predict_text(text)
        status = "positive" if pro > 0.5 else "negative"
        pro = (1 - pro) if status == "negative" else pro
        print(f'Predicted sentiment is {status} with a probability of {pro}')
        return JSONResponse(content= {
            "sentiment": status,
            "probability": pro
        })
    except Exception as e:
        raise SentimentException(e, sys)

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)