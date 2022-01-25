# iteration3
Seoul Air Quality Improvement Proposal
Using data mining to predict the value of PM2.5
ITERATION 3 OSAS
Open Source Analytics Solution – Python, Jupyter, MySQL, MySQL 
Workbench, Kettle/Spoon, Tableau, & Weka. 

Abstract
This project is applying python as a programming language and some libraries such as
Panda, Matplotlib and sklearn to solve the problem of Seoul air pollution with the help of
data mining. The data mining objectives including building a model to predict the value of
PM2.5 concentration. Providing timely and accurate forecast of PM2.5 to the public and
take emission control actions to achieve the business goal of decreasing the value of
PM2.5. 

The dataset includes air pollution measurement information in Seoul, which is collected
from website: https://www.kaggle.com/bappekim/air-pollution-in-seoul.
The original data is provided by The Seoul Metropolitan City Institute of Health and
Environment as below:
https://data.seoul.go.kr/dataList/OA-15526/S/1/datasetView.do
https://data.seoul.go.kr/dataList/OA-15516/S/1/datasetView.do
https://data.seoul.go.kr/dataList/OA-15515/S/1/datasetView.do
This is air pollution measurement information provided by the air pollution measurement network
system of the Seoul Institute of Health and Environment. The 1-hour average measurement is
provided at 5 minutes every hour after calibration. It provides faster data by directly calibrating the
measured results at the measurement stations in each autonomous district and providing them
directly.
Meter status: "0": Normal "1": Calibration, "2": Abnormal, "4": Power cut off, "8": Under maintenance,
"9": Data error,
exceeding national standards Category: "0": Normal , “1”: Exceeding
local government standards Category: “0”: Normal, “1”: Exceeding
Seoul Metropolitan Air Pollution Monitoring Station information.
It provides information such as measurement place name, measurement place address, and
installation location.
Seoul Metropolitan Air Pollution Measurement Items Information
This is information on the measurement items of Seoul's air pollution network.
It provides measurement item name, communication symbol, unit, order, number of decimal
places, and legend information.
The original data is saved as one single file by year.
The dataset used for this project combined information from 3 original data files.
The dataset in this project contains several data, including: 
16
Measurement data. The date (in the format of yyyy-mm-dd) and time (in the format of hhmm-ss) indicates the 1-hour average for measurement value. The interval is one hour.
Station code. There are 25 different measurement stations in Seoul that provide
measurement data. The station code is start from 101 to 125. Each station code represents
a station.
Latitude. The latitude of stations is recorded.
Longitude. The longitude of stations.
SO2. sulfur dioxide (SO2) concentration. The 1-hour average measurement is provided at 5 minutes every
hour after calibration.
NO2. nitrogen dioxide (NO2) concentration. The 1-hour average measurement is provided at 5 minutes
every hour after calibration.
O3. ozone (O3) concentration. The 1-hour average measurement is provided at 5 minutes every hour after
calibration.
CO. CO concentration.
PM10. The concentration of particular matter with a diameter of 10 microns or less.
PM2.5. The concentration of particular matter with a diameter of 2.5 microns or less. 
