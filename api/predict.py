import pandas as pd
from api.model_loader import model,feature_columns
def predict_price(data):
 frame=pd.DataFrame([data]).reindex(columns=feature_columns,fill_value=0)
 return max(0.0,float(model.predict(frame)[0]))