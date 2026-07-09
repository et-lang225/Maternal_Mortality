# Maternal Mortality
#### I will be using Synthea data to predict maternal mortality from a number of different factors. This README file will be updated as I make progress.
#### The processing of the database files could be completed entirely within pandas, but I wanted to simulate a traditional database extraction in order to present an example pipeline.
#### The most feasible way to do this right now with the processing power available to me is to query 100K patients from Synthea
#### So far have code to run a Naive Bayes classifier, a Logisitic Regression Cross Validation (for EN regularization hyperparameters), A Gaussian SVM classifer, a cross-validation loop for a Random Forest classifier, a cross-validation loop for a gradient boosting classifier, and a CV loop for a PyTorch Neural Network classifier
#### I will compare the final model runs with area under a precision recall curve (true positives will be the hardest to come by) and generate graphics of model results

### Here is how I generated data from Synthea
### First, go to ./src/main/resources/synthea.properties and make sure exporter.csv.export = true.
``` bash
git clone https://github.com/synthetichealth/synthea.git         
cd synthea         
.\gradlew.bat build check test        
.\run_synthea -p 100000 -g F "Louisiana"
```         

Note: Might have to install the most recent version of java in order to run synthea. I used a network java folder and ran java from that folder.
```bash          
Invoke-WebRequest -Uri "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip" -OutFile "jdk17.zip"         
Expand-Archive -Path "jdk17.zip" -DestinationPath ".\jdk17"         
$env:JAVA_HOME = "$(Get-Location)\jdk17\jdk-17.0.11+9"          
$env:Path = "$env:JAVA_HOME\bin;" + $env:Path        
java -version         
```