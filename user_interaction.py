#this is the file used to make this project usable by non coders and get the output of the Model 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#lets build the class that will do set of operation like 
# getting the data, doing the featuring ,training the data and finally displaying the data in the user friendly manner 

class GenderDetection :
    def __init__(self):
        self.data = None
        self.count = 0
        pass
    def get_data(self):
        try:
            #collecting the type of input for easy segrigation
            print("\n\nWhich type of input would you like?\n")
            print("1.Single Name\n2.List of names\n3.CSV/Excel file")
            choice = input("\nYour choice:\t")

            print("\n\t----------------------------------\t\n") #just to make sure proper spacing 

            self.fetch_file(choice)
        except Exception as e:
            print(e)

    def fetch_file(self, choice):
        try:

            #removing the balck space and making it a integer
            choice = choice.strip()
            choice = int(choice)

            print(f"Your input is {choice}\n")
            if choice == 1:
                #this should take the input of single name
                self.data = input("\nEnter the name:\t")
               
            elif choice == 2:
                #this should take the list and convert into a DataFrame
                final_list =[]

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
                
                pass
            else:
                print(f"!!! {choice} is not the valid input !!!")
                print("Plese try again")
                self.count = self.count+1
                if self.count >=3:
                    print("\n!!!\tYou have exceded the repeat limit, Try again later\t!!!\n")
                    print("\t------Thank you ------\n")
                    exit()
                else:
                    self.get_data()

        except Exception as e:
            print(e)

    def convert_into_feature(self):
        pass

    def training(self):
        pass
    
    def output_printing(self):
        pass

def main():
    Detector = GenderDetection()
    Detector.get_data()
    Detector.convert_into_feature()
    Detector.training()
    Detector.output_printing()

if __name__ == "__main__":
    main()