from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from api.schema import PricePredictionRequest
from api.model_loader import category_encoder,feature_columns
from api.predict import predict_price
ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title="PricePilot AI",version="2.0.0",description="Olist product price estimation API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
@app.get("/",include_in_schema=False)
def dashboard(): return FileResponse(ROOT/"index.html")
@app.get("/health")
def health(): return {"status":"ok","model":"XGBoost","features":len(feature_columns)}
@app.get("/metadata")
def metadata(): return {"categories":sorted(str(x) for x in category_encoder.classes_),"currency":"BRL"}
@app.post("/predict-price")
def predict(request:PricePredictionRequest):
 try: return {"predicted_price":round(predict_price(request.model_dump()),2),"currency":"BRL","model":"XGBoost"}
 except Exception as exc: raise HTTPException(status_code=422,detail=f"Prediction failed: {exc}") from exc