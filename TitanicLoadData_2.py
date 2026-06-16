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