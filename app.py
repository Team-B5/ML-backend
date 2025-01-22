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

#CORS handling is one of the first things if different parts of projects are deployed separately.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#the purpose of this get request is to ensure server is up and running
@app.get('/')
def read_data():
    print("request recieved")
    return {"message": "This is a simple GET request to the server which works as a sanity check that the API is up and running. If somehow you landed up here but want to explore our entire project make a post request to '/test' url." 
    }


#the actual request which accepts the data, calculates the prediction and returns it.
@app.post('/test')
async def predict_air(data: Test):
    # print(data)
    newdata = data.model_dump()
    Temperature = newdata["Temperature"]
    NO2 = newdata["NO2"]
    SO2 = newdata["SO2"]
    CO = newdata["CO"]
    Proximity_to_Industrial_Areas = newdata["Proximity_to_Industrial_Areas"]
    #print(Temperature, NO2, SO2, CO, Proximity_to_Industrial_Areas)
    #print(classifier.predict([[Temperature, NO2, SO2, CO, Proximity_to_Industrial_Areas]]))
    answer = classifier.predict([[Temperature, NO2, SO2, CO, Proximity_to_Industrial_Areas]])
    #print(answer, "  the predictions is")

    #we configured our model labels as 0-Poor, 1-Good
    if answer == 0:
        return "Poor"
    else:
        return "Good"


if __name__ == "main":
    uvicorn.run(app, host="0.0.0.0", port=8000)