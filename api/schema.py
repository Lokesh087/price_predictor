from pydantic import BaseModel


class PricePredictionRequest(BaseModel):

    product_category_name: int

    freight_value: float

    product_weight_g: float

    product_length_cm: float

    product_height_cm: float

    product_width_cm: float

    purchase_month: int

    purchase_day: int

    purchase_hour: int

    purchase_weekday: int

    purchase_quarter: int

    is_weekend: int

    delivery_days: int

    estimated_delivery_days: int

    delivery_difference: int

    product_volume: float

    product_density: float