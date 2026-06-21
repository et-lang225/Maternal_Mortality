# Maternal Mortality
#### I will be using Synthea data to predict maternal mortality from a number of different factors. This README file will be updated as I make progress.
#### So far I have only started simulating data and converting csv files to a SQL database.
#### The processing of the database files could be completed entirely within pandas, but I wanted to simulate a traditional database extraction in order to present an example pipeline.
#### So far constructed the bones of a PyTorch Neural Network classifier, a cross-validation loop for a Random Forest classifier, and a Logisitic Regression Cross Validation (for EN regularization hyperparameters)
#### I will have to establish cross validation for the NN model, code a Naive Bayes model, and code a cross validation loop for a XGBoost model
#### I have developed this in no particular order, but probably should have started with the simpler Naive Bayes and Logistic Regression before moving to the bagging and boosting models, then to a neural network

### Here is how I generated data from Synthea
``` bash
git clone https://github.com/synthetichealth/synthea.git         
cd synthea         
.\gradlew.bat build check test        
exporter.csv.export = true        
.\run_synthea -p 1000 -g F "Louisiana"
```         

Note: Might have to install the most recent version of java in order to run synthea. I had to install a local java folder and inform the powershell to run java from that folder.
```bash          
Invoke-WebRequest -Uri "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip" -OutFile "jdk17.zip"         
Expand-Archive -Path "jdk17.zip" -DestinationPath ".\jdk17"         
$env:JAVA_HOME = "$(Get-Location)\jdk17\jdk-17.0.11+9"          
$env:Path = "$env:JAVA_HOME\bin;" + $env:Path        
java -version         
```