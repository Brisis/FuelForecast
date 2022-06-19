import petrol_forecast
import diesel_forecast


def getForecast(months: int, fuel_type: str):
    if fuel_type == "Petrol":
        petrol_forecast.forecastPetrol(months-1)
    else:
        diesel_forecast.forecastDiesel(months-1)
