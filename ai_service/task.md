# Description

AI service for the flood forecasting system.

The AI service need to return unindention map of the next 5 minutes, 30 minutes and 2 hours.

# Input
We will have a data stream (rainfall) that constantly stream through our system (lets say we can control the frequency of this for the MVP version to demo)

The model will receive input and using past prediction data with DEM and Manning map as input

# Output
The output will be a map of the flood depth in the next 5 minutes, 30 minutes and 2 hours.

# Input data for render service
+ metadata.json
+ flood_depths.npy (or image and send through base 64| or consider a way to send the data in a more efficient way) 

# More information
[metadata](exp/metadata.md)

# experiment and model
[train](exp/train.py)
[inference](exp/inference.py)
[best model](weights/best_model.pth)
# Static File Locations
static_data\

# rainfall stream
stored in rainfall\*.tif
should be send through JSON containing a base64 image

# Initial conditions
![initial condition](static_data/Australia_initial_condition.tif)
Use this if found no predicted stream of flood depth

# TASK
+ Create an Data Stream service that send data about rainfall (30 minutes interval in real time, but for the demo, lets say I define a "T" in .env that control the frequency of this data stream (which mean after T seconds, the service will send a new 30-minute-interval data for AI service))
+ AI service will receive the data and predict the flood depth in the next 5 minutes, 30 minutes and 2 hours. (use input as said)
+ Database service - store data about the rainfall and the flood depth prediction

# API endpoint for AI service
+ return the calculated and stored (latest) flood depth prediction in the next 5 minutes, 30 minutes and 2 hours. (get with parameter "time_horizon" with value "5", "30", "240")

# Folder to work with
D:\project\d4l\ai_service
