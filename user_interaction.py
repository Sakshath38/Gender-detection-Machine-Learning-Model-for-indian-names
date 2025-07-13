#this is the file used to make this project usable by non coders and get the output of the Model 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os
import sys
import re

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
            self.choice = int(input("\nYour choice:\t"))

            if self.choice == 1 or self.choice == 2 or self.choice == 3:
                print("\n\t----------------------------------\t\n") #just to make sure proper spacing
                print(f"Your input is {self.choice}\n") 
                self.fetch_file(self.choice)
            else:
                print(f"!!! {self.choice} is not the valid input !!!")
                self.count = self.count +1
                if self.count >=3:
                    print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                    print("\t------Thank you ------\n")
                    sys.exit(1)
                else:
                    print("-----\tPlease try again\t-----")
                    self.get_data() #redirected to this


        except Exception as e:
            #mostly for not integer entries this will be used
            print(f"\n!!!\tThere was an error : {e}\t!!!\n")
            self.count = self.count +1
            if self.count >=3:
                print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                print("\t------Thank you ------\n")
                sys.exit(1)
            else:
                print("\n-----\tPlease try again\t-----")
                self.get_data() #redirected to this

    def fetch_file(self, choice):

        try:

            final_list =[]
            if choice == 1:
                #this should take the input of single name
                value = input("\nEnter the name:\t") 
                name = value.lower().strip()
                cleaned_name = self.clean_name(name= name) #this will clean the name

                if self.is_valid_name(cleaned_name): #this will only append the right valid names
                    final_list.append(cleaned_name)
                else:
                    print(f"\nWarning: '{name}' is not a valid name\n")
                    self.count = self.count +1
                    if self.count >=3:
                        print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                        print("\t------Thank you ------\n")
                        sys.exit(1)
                    else:
                        print("-----\tPlease try again\t-----")
                        self.fetch_file(1) #this will take back to the loop and start again with choice =1
                        return

                self.data = pd.DataFrame({
                    "name":final_list
                })
                #convert single name as a list of single element
               
            elif choice == 2:
                #this should take the list and convert into a DataFrame

                print("(Note: use coma(,) to separate between names)")
                print("\nEnter the list of Names below \t")
                user_input = input("Names:\t")

                #splitting and removing space in the input 
                name_list = [name.strip().lower() for name in user_input.split(",")]
                
                #inputing the clean and filtered data into the list
                for name in name_list:
                    cleaned_name = self.clean_name(name= name) #this will clean the name

                    if self.is_valid_name(cleaned_name): #this will only append the right valid names
                        final_list.append(cleaned_name)
                    else:
                        print(f"\nWarning: '{name}' is not a valid name and was skipped\n")
                
                #check if the final final_list is empty or not
                if not final_list:
                    print("Error: No valid names were provided!\n")
                    self.count = self.count +1
                    if self.count >=3:
                        print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                        print("\t------Thank you ------\n")
                        sys.exit(1)
                    else:
                        print("-----\tPlease try again\t-----")
                        self.fetch_file(2) #this will take back to the loop and start again with choice =2
                        return
                    
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
                path = path.strip('"')
                extention = self.get_file_extension(path)
                
                try:
                    if extention == ".csv":
                        self.data = pd.read_csv(path)
                        
                    elif extention == ".xlsx":
                        self.data = pd.read_excel(path)
                        
                    else:
                        if extention != ".xlsx" or extention != ".csv":
                            print(f"\n\t[{extention}] cannot be passed\n") 
                        self.count = self.count +1
                        if self.count >=4:
                            print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                            print("\t------Thank you ------\n")
                            sys.exit(1)
                        else:
                            print("-----\tPlease try again\t-----")
                            self.fetch_file(3) #this will take back to the loop and start again with choice =3
                            return
                        
                except FileNotFoundError:
                    print(f"\n!!!\t[{path}] is not accessible\n")
                    self.count = self.count +1
                    if self.count >=4:
                        print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                        print("\t------Thank you ------\n")
                        sys.exit(1)
                    else:
                        print("-----\tPlease try again\t-----")
                        self.fetch_file(3) #this will take back to the loop and start again with choice =3
                        return
                    
                
                #get the clean names for excel as other files 

                #local function to get the variable
                def getting_column_name(data=self.data):
                    name_column = input("\nEnter the exact column name with names:\t")
                    name_column = name_column.strip().strip('"') #this will help in removing space 

                    if name_column not in data.columns:
                        #The only case of failure is at choice == 3 
                        print(f"'{name_column}' is not available in the sheet\n")

                        #give one more option to enter the column name
                        self.count = self.count +1
                        if self.count >=3:
                            print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                            print("\t------Thank you ------\n")
                            sys.exit(1)
                        else:
                            print("-----\tPlease try again\t-----")
                            return getting_column_name() #this will redirect into same function
                            
                    else:
                        #load the data otherwise
                        return data[name_column]
                     
                 
                df = getting_column_name() #this will load the exact data from the dataframe

                rejected_names = [] #this will store the rejected names from data frame

                #now checking the names in the df for validity
                for name in df:
                    cleaned_name = self.clean_name(name)
                    
                    if self.is_valid_name(cleaned_name): #this will only append the right valid names
                        final_list.append(cleaned_name)
                    else:
                        rejected_names.append(name)
                    
                if rejected_names:
                    print(f"\n!!!\tWarning below name(s) are rejeted\t!!!\n")
                    print(rejected_names)
                    

                if not final_list:
                    print("Error: No valid names were provided!\n")
                    self.count = self.count +1
                    if self.count >=3:
                        print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                        print("\t------Thank you ------\n")
                        sys.exit(1)
                    else:
                        print("-----\tPlease try again\t-----")
                        self.fetch_file(2) #this will take back to the loop and start again with choice =2
                        return

                #converting the data into the required form 
                self.data = pd.DataFrame({
                    "name" : final_list
                })      

            #only excecute this if there is data
            if hasattr(self,'data') and self.data is not None and not self.data.empty:
                print("\n------------\tFinal DataFrame\t------------")
                print(self.data)
                    
                #end point of this elif ladder so end and print
                print("\n\t----------------------------------\t\n") #just to make sure proper spacing 
                    
                self.convert_into_feature(self.data) #this will invoke the next function by default

           
        except Exception as e:
            print(f"\n!!!\tThere was an error : {e}\t!!!\n")
            

    #to get the relevant names only      
    def clean_name(self,name):

        if not name or pd.isna(name) or name is None:
            return""
        
        try:
            #handle numerical inputs
            cleaned = str(name).lower().strip()
            cleaned = re.sub(r'[^a-zA-Z\s]','',cleaned)
            # cleaned = ' '.join(cleaned.split()) #this will return full name
            
            if cleaned.strip(): #only if it is not empty
                cleaned = cleaned.split()[0] #this will only add the first letter

            return cleaned.strip()
        
        except Exception as e:

            print(f"\n!!!\tThere was an error : {e}\t!!!\n")
    
    #this will help in checking the validity of the name 
    def is_valid_name(self,name):
        if not name or not isinstance(name, str):
            return False

        name = name.strip()

        if not re.match(r'^[a-zA-Z\s]+$', name):
            return False
        
        if len(name)<2:
            return False
        
        if name =="":
            return False

        return True
   
    def get_file_extension(self,filename):
        name,extention = os.path.splitext(filename)
        return extention.lower()
    
    
    def convert_into_feature(self,df,name_col='name'):

        try:
            print("Extracting features .........\n")
            df_copy = df.copy()
            feature = self.extract_name_features(df_copy) #this will work for all choices as the data is standardized to column_name = "name"
            print("\n------\tYour features are as below:\t------\n")    
            print(feature)    #this will print the feature

            self.training(feature) #invoke the training set

            return feature
             
        except Exception as e:
            
            print(f"\n!!!\tThere was an error : {e}\t!!!\n")
            
            
    
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

    def training(self,features):

        pass
    
    def output_printing(self):
        pass

def main():
    try:
        Detector = GenderDetection()
        Detector.get_data()
        Detector.output_printing()

    except Exception as e:
        print(f"\n!!!\tThere was an error : {e}\t!!!\n")        


if __name__ == "__main__":
    main()