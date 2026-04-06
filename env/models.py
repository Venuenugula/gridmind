from pydantic import BaseModel, Field


class Observation(BaseModel):
    demand: float
    renewable_supply: float
    thermal_capacity: float
    battery_storage: float

    price_market: float
    outage_risk: float

    weather_factor: float
    time_of_day: int
    day_of_week: int


class Action(BaseModel):
    buy_power: float = Field(ge=0)
    use_thermal: float = Field(ge=0)
    charge_battery: float = Field(ge=0)
    discharge_battery: float = Field(ge=0)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}