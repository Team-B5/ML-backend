import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

pickle_in = open("final_random_forest_model.pkl", "rb")
classifier = pickle.load(pickle_in)

class Test(BaseModel):
    Temperature :float
    NO2:float
    SO2:float
    CO:float
    Proximity_to_Industrial_Areas:float

app = FastAPI()

origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_data():
    print("request recieved")
    return {"message": "hello world"}

# @app.post('/predict')
# def predict_air(data: dict):
#     print(data)
#     print(data["CO"])
#     return {"CO" : data["CO"]}

@app.post('/test')
async def predict_air(data: Test):
    # print(data)
    newdata = data.model_dump()
    Temperature = newdata["Temperature"]
    NO2 = newdata["NO2"]
    SO2 = newdata["SO2"]
    CO = newdata["CO"]
    Proximity_to_Industrial_Areas = newdata["Proximity_to_Industrial_Areas"]
    print(Temperature, NO2, SO2, CO, Proximity_to_Industrial_Areas)
    print(classifier.predict([[Temperature, NO2, SO2, CO, Proximity_to_Industrial_Areas]]))
    answer = classifier.predict([[Temperature, NO2, SO2, CO, Proximity_to_Industrial_Areas]])
    print(answer, "  the predictions is")
    if answer == 0:
        return "Poor"
    else:
        return "Good"
    # print(data["CO"])
    # return 0


if __name__ == "main":
    uvicorn.run(app, host="0.0.0.0", port=8000)