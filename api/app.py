from fastapi import FastAPI

from api.schema import PricePredictionRequest

from api.predict import predict_price


app = FastAPI(

    title="PricePilot AI",

    version="1.0"

)


@app.get("/")

def home():

    return {

        "message":

        "PricePilot AI API Running"

    }


@app.post("/predict-price")

def predict(

    request: PricePredictionRequest

):

    prediction = predict_price(

        request.dict()

    )

    return {

        "Predicted Price":

        round(

            prediction,

            2

        )

    }