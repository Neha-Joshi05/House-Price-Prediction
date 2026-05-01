"""
schemas.py
----------
Pydantic models for request / response validation.
"""

from pydantic import BaseModel, Field
from typing import Literal


class HouseFeatures(BaseModel):
    area_sqft:       float = Field(..., gt=100,  description="Built-up area in sq ft")
    bedrooms:        int   = Field(..., ge=1, le=10)
    bathrooms:       int   = Field(..., ge=1, le=5)
    floors:          int   = Field(..., ge=1, le=5)
    age_years:       float = Field(..., ge=0)
    garage:          int   = Field(..., ge=0, le=1)
    pool:            int   = Field(..., ge=0, le=1)
    furnished:       Literal["unfurnished", "semi-furnished", "furnished"] = "unfurnished"
    location_tier:   int   = Field(..., ge=1, le=4, description="1=Premium, 4=Budget")
    distance_cbd_km: float = Field(..., gt=0)
    school_rating:   float = Field(..., ge=1, le=10)
    crime_index:     float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "area_sqft":       2200,
                "bedrooms":        3,
                "bathrooms":       2,
                "floors":          2,
                "age_years":       8,
                "garage":          1,
                "pool":            0,
                "furnished":       "semi-furnished",
                "location_tier":   2,
                "distance_cbd_km": 12.5,
                "school_rating":   7.5,
                "crime_index":     25.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price:  float
    model_used:       str
    confidence_range: dict
    input_summary:    dict