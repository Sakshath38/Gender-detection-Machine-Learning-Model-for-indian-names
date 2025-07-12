#this is the file used to make this project usable by non coders and get the output of the Model 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os

#lets build the class that will do set of operation like 
# getting the data, doing the featuring ,training the data and finally displaying the data in the user friendly manner 

class GenderDetection :
    def __init__(self):
        self.data = None
        self.count = 0
        self.choice = None
        
    def get_data(self):
        try:
            #collecting the type of input for easy segrigation
            print("\n\nWhich type of input would you like?\n")
            print("1.Single Name\n2.List of names\n3.CSV/Excel file")
            self.choice = input("\nYour choice:\t")

            print("\n\t----------------------------------\t\n") #just to make sure proper spacing 

            self.fetch_file(self.choice)
        except Exception as e:
            print(e)

    def fetch_file(self, choice):

        final_list =[]
        try:

            #removing the balck space and making it a integer
            choice = choice.strip()
            choice = int(choice)

            print(f"Your input is {choice}\n")
            if choice == 1:
                #this should take the input of single name
                value = input("\nEnter the name:\t") 

                value = value.lower().strip()
                final_list.append(value)

                self.data = pd.DataFrame({
                    "name":final_list
                })
                #convert single name as a list of single element
               
            elif choice == 2:
                #this should take the list and convert into a DataFrame

                print("(Note:use coma(,) to separate between names)")
                print("\nEnter the list of Names below \t")
                list = input("Names:\t")

                #splitting the input of the list into multiple strings 
                list = list.split(",")
                
                #inputing the clean and filtered data into the list
                for i in range(len(list)):
                    list[i] = list[i].lower().strip()
                    final_list.append(list[i])
                    
                print(f"\nYour list of names:\t{final_list}\n")

                #converting to dataframes

                self.data = pd.DataFrame({
                    "name":final_list
                })
                
            elif choice == 3:
                #directly read the files as the panda input after checking its .csv or excel
                print("\n !!!\tNote : Only paste the Excel or CSV files\t!!!\n")

                print(r'Example:C:\Users\....\Gender-detection-Machine-Learning-Model-for-indian-names\Data_files\Indian_firstname_Gender_Data.csv')
                
                path = str(input("\n Enter the path to the file/directory below :\n"))
                extention = self.get_file_extension(path)
                
                if extention == ".csv":
                    self.data = pd.read_csv(path)
                    
                elif extention == ".xlsx":
                    self.data = pd.read_excel(path)

                else:
                    print(f"\n!!!\t[{path}] is not accessible\n")
                    print("-----\tPlease try again\t-----")
                    self.count = self.count +1
                    if self.count >=3:
                        print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                        print("\t------Thank you ------\n")
                        exit()
                    else:
                        self.fetch_file("3") #this will take back to the loop and start again with choice =3
                
                print("\n")
                print(self.data)
                #end point of this elif ladder so end and print
                print("\n\t----------------------------------\t\n") #just to make sure proper spacing 
                
                
            else:
                print(f"!!! {choice} is not the valid input !!!")
                print("-----\tPlease try again\t-----")
                self.count = self.count+1
                if self.count >=3:
                    print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                    print("\t------Thank you ------\n")
                    exit()
                else:
                    self.get_data()

            self.convert_into_feature(choice = choice) #this will invoke the next function by default
        except Exception as e:
            print(e)
    
    def get_file_extension(self,filename):
        name,extention = os.path.splitext(filename)
        return extention.lower()
    
    def convert_into_feature(self, choice):

        try:
            if choice == 1:
                feature = self.extract_name_features(self.data)
                
            elif choice == 2:
                feature = self.extract_name_features(self.data)
                
            elif choice == 3:
                name_column = input("\nEnter the exact column name with names:\t")
                name_column = name_column.strip() #this will help in removing space 

                feature = self.extract_name_features(self.data, name_col=name_column)
                
            else:
                print('\nUnable to get features\n')
                print("-----\tPlease try again\t-----")
                print("------\tThank you\t------\n")

            print("\n------\tYour features are as below:\t------\n")    
            print(f"{feature}")    #this will print the feature
            return feature
            
        except Exception as e:
            print(e)
    
    #this is the function for getting the features
    def extract_name_features(self,df, name_col ='name'):
        df['name'] = df[name_col].str.strip().str.lower()
        df["name_length"] = df["name"].apply(len)
        df["first_letter"] = df["name"].str[0]
        df["last_letter"] = df["name"].str[-1]
        df["vowel_count"] = df["name"].apply(lambda x: sum(1 for c in x if c in "aeiou"))
        df["consonant_count"] = df["name_length"] - df["vowel_count"]
        df["suffix_2"] = df["name"].str[-2:]
        df["prefix_2"] = df["name"].str[:2]
        df["is_last_letter_a"] = (df["last_letter"]== "a").astype(int)

        #rearranging the df accordingly 
        df_function = df[["name","name_length","first_letter","last_letter","vowel_count",	"consonant_count",	"suffix_2","prefix_2","is_last_letter_a"]].copy()
        return df_function

    def training(self):
        pass
    
    def output_printing(self):
        pass

def main():
    Detector = GenderDetection()
    Detector.get_data()
    Detector.training()
    Detector.output_printing()
    

if __name__ == "__main__":
    main()