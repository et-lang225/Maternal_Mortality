# Maternal Mortality
#### I will be using Synthea data to predict maternal mortality from a number of different factors. This README file will be updated as I make progress.
#### So far I have only started simulating data and converting csv files to a SQL database.
#### The processing of the database files could be completed entirely within pandas, but I wanted to simulate a traditional database extraction in order to present an example pipeline.
#### I am working on getting enough data (1million records) to split data into a train and test set
#### I queried 1,000 total patients first but that only yeilded about 200 pregnant women and 5 maternal mortality cases. I will have to be vigilant about model evaluation metrics to be sure that True Positives are being classified correctly (possibly focusing on Precision and Recall). 
#### So far have code to run a Naive Bayes classifier, a Logisitic Regression Cross Validation (for EN regularization hyperparameters), A Gaussian SVM classifer, a cross-validation loop for a Random Forest classifier, a cross-validation loop for a gradient boosting classifier, and the bones of a PyTorch Neural Network classifier
#### Need to include a cross validation loop for the NN model 

### Here is how I generated data from Synthea
### First, go to ./src/main/resources/synthea.properties and make sure exporter.csv.export = true.
``` bash
git clone https://github.com/synthetichealth/synthea.git         
cd synthea         
.\gradlew.bat build check test        
.\run_synthea -p 1000000 -g F "Louisiana"
```         

Note: Might have to install the most recent version of java in order to run synthea. I had to install a local java folder and inform the powershell to run java from that folder.
```bash          
Invoke-WebRequest -Uri "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip" -OutFile "jdk17.zip"         
Expand-Archive -Path "jdk17.zip" -DestinationPath ".\jdk17"         
$env:JAVA_HOME = "$(Get-Location)\jdk17\jdk-17.0.11+9"          
$env:Path = "$env:JAVA_HOME\bin;" + $env:Path        
java -version         
```