import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#--------------------------------------------------------------------------------
# Function Name : DisplayInfo
# Description : It displays the formated titel
# Parameters :  title(str)
# Return :      None
# Date :        14/3/2026
# Author :      Rukmini Gaikwad
#--------------------------------------------------------------------------------

def DisplayInfo(title):
    print("\n"+ "="*70)
    print(title)
    print("="*70)


#--------------------------------------------------------------------------------
# Function Name : ShowData
# Description : It shows basic information about dataset
# Parameters :  df
#               df ->  pandas dataframe object
#               message
#               message -> Headinfg text to display
# Return :      None
# Date :        14/3/2026
# Author :      Rukmini Gaikwad
#--------------------------------------------------------------------------------
def ShowData(df,message):
    DisplayInfo(message)

    print("First 5 rows of dataset")
    print(df.head())

    print("\nShape of dataset")
    print(df.shape)

    print("\nColumns name :")
    print(df.columns.tolist())

    print("\nMisssing values in each column")
    print(df.isnull().sum())


#--------------------------------------------------------------------------------
# Function Name : CleanTitanicData
# Description : It does preprocessing 
#               It removes unnecessary columns
#               It handles missing value
#               It converts text data to numeric format
#               It does encoding to categorical 
# Parameters :  df -> pandas dataframe
# Return    :   df-> clean pandas dataframe
# Date :        14/3/2026
# Author :      Rukmini Gaikwad
#--------------------------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original data")
    print(df.head())

    #Remove Unnecessary column
    drop_columns =["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n Columns to be dropped : ")
    print(existing_columns)

    #drop the unwanted columns
    df = df.drop(columns = existing_columns)
    DisplayInfo("Step 2 : Data after columns removal")
    print(df.head())

    #Hnadle age column
    if "Age" in  df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        #coerce -> Invalid values gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce") 

        age_median = df["Age"].median()

        # Replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge column after preprocessing")
        print(df["Age"].head(10))

    # Handle fare column 
    if "Fare" in df.columns:
        print("\n Fare columns before preprocessing")    
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce") 

        fare_median = df["fare"].median()
        print("\n Median of fare colun=mn is : ",fare_median)

        # Replace missing values with median
        df["Fare"] = df["Fare"].fillna(fare_median)

        print("\nFare column after preprocessing")
        print(df["Fare"].head(10))

       #Handle embarked column
    if "Embarked" in df.columns:
        print("\n Embarked columns before preprocessing")    
        print(df["Embarked"].head(10))

        #Convert the data into string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        #Remove missing value
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

        # Get most frequent value -> ex - tourist
        embarked_mode = df["Embarked"].mode()[0]
        print("\nMode of embarked column :",embarked_mode)

        df["Embarked"] = df["Embeked"].fillna(embarked_mode)

        print("\n Embarked columns before preprocessing")    
        print(df["Embarked"].head(10))


    # Hndle Sex column
    if "Sex" in df.columns:
        print("\n  Sex columns before preprocessing")    
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce") 

        print("\n Sex columns before preprocessing")    
        print(df["Sex"].head(10))

    DisplayInfo("Data after preprocessing")    
    print(df.head())

    print("\nMissing value after preprocessing")
    print(df.isnull().sum())

    

    return df 


#--------------------------------------------------------------------------------
# Function Name : MarvellousTitanicLogistic
# Description : This is maion pipeline controller
#               It loads the datset,shows raw data
#               It preprocess the dataset and train the model
# Parameters :  Data path of dataset file
# Return :      None
# Date :        14/3/2026
# Author :      Rukmini Gaikwad
#--------------------------------------------------------------------------------



def MarvellousTitanicLogistic(DataPath):
    DisplayInfo("Step 1 : Loading the dataset")
    df = pd.read_csv(DataPath)

    ShowData(df,"Initial dataset")

    df = CleanTitanicData(df)




#--------------------------------------------------------------------------------
# Function Name : main
# Description :   Starting point of the application
# Parameters :    None
# Return :        None
# Date :          14/3/2026
# Author :        Rukmini Gaikwad
#--------------------------------------------------------------------------------

def  main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")
    


if __name__ == "__main__":
    main()