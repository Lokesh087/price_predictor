import pandas as pd

from api.model_loader import (

    model,

    feature_columns

)


def predict_price(data):

    df = pd.DataFrame(

        [data]

    )

    df = df.reindex(

        columns=feature_columns,

        fill_value=0

    )

    prediction = model.predict(df)

    return float(prediction[0])