#this is the file used to make this project usable by non coders and get the output of the Model
import pandas as pd
import numpy as np

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
        self.model = None
        self.sheet = None
        self.prediction = None
    
    def counter(self):
        self.count = self.count +1
        if self.count >=3:
            print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
            print("\t------Thank you ------\n")
            sys.exit(1) #this is used to quit the program once the limits are hit
        else:
            print("-----\tPlease try again\t-----")
        
        return None
    

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
                self.counter()
                self.get_data() #redirected to this


        except Exception as e:
            #mostly for not integer entries this will be used
            print(f"\n!!!\tThere was an error : {e}\t!!!\n")
            self.counter()
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
                    self.counter()
                    self.fetch_file(1) #this will take back to the loop and start again with choice =1
                    
                    return None   

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
                    self.counter()
                    self.fetch_file(2) #this will take back to the loop and start again with choice =2
                    return None
                    
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
                self.path = path.strip('"')
                extention = self.get_file_extension(path)
                
                try:
                    if extention == ".csv":
                        self.data = pd.read_csv(path)
                        
                        '''require to the actual sheet during gender test, for checking accuracy in test_module'''
                        self.sheet = self.data.copy()
                        
                    elif extention == ".xlsx":
                        self.data = pd.read_excel(path)
                        self.sheet = self.data.copy()
                        
                    else:
                        if extention != ".xlsx" or extention != ".csv":
                            print(f"\n\t[{extention}] cannot be passed\n") 
                        self.counter()
                        self.fetch_file(3) #this will take back to the loop and start again with choice =3
                        return None
                        
                except FileNotFoundError:
                    print(f"\n!!!\t[{path}] is not accessible\n")
                    self.counter()
                    self.fetch_file(3) #this will take back to the loop and start again with choice =3
                    return None
                    
                
                #get the clean names for excel as other files 

                #local function to get the variable
                def getting_column_name(data=self.data):
                    name_column = input("\nEnter the exact column name with names:\t")
                    name_column = name_column.strip().strip('"') #this will help in removing space 

                    if name_column not in data.columns:
                        #The only case of failure is at choice == 3 
                        print(f"'{name_column}' is not available in the sheet\n")

                        #give one more option to enter the column name
                        self.counter()
                        return getting_column_name() #this will redirect into same function
                            
                    else:
                        #we need to generalise the name throughout
                        self.sheet.rename(columns={name_column:"name"}, inplace= True)

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
                    self.counter()
                    self.fetch_file(3) #this will take back to the loop and start again with choice =2
                    return None

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
            return ""
        
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

            self.test_model(feature) #to use the trained model

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


    def test_model(self,features):
        
        '''from model''' 
        #defined localy because it should only be used when called for!
        import model
        from model import MODEL_FILE,PIPELINE_FILE
        import joblib


        if not os.path.exists(MODEL_FILE):
            model()
        
        #INFERENCE PHASE 
        print("\nRunning model for prediction ......\n")
        model_trained = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        
        #now we need to get the data for prediction
        '''Connect the user data for prediction'''
        if features is None or features.empty or features.isnull().any().any() :
            df = pd.read_csv("model.help/X_test.csv")
            print(f"\n!!!\tUsing internal 'test' set as the input feature is EMPTY\t!!!\n")
            X_test = True
        else:
            #this is the structured input to the model 
            df = features 


        '''Testing the model'''

        transformed_features = pipeline.transform(df)
        self.prediction = model_trained.predict(transformed_features)
        df["prediction"] = self.prediction

        df.to_csv("output.csv", index = False)
        print("\n\t\t .... Inference completed succefully....\n")
        print("Results saved to 'output.csv'\n")
        
        #displaying the output
        self.output_printing(predict_input=self.prediction, feature_name=df["name"])
        
        #checking for accuracy 
        accuracy = input("\n\t Do you want to check accuracy? (Yes/No) :")
        accuracy = accuracy.strip().lower()

        if accuracy == "yes":

            def gender_cleaned_list(data,gender_column, return_value):
                
                final_list = []

                #since 3 is the sheet, it will through an error
                if return_value != 3:
                    name_list = [name.strip().lower() for name in gender_column.split(",")]
                else:
                    name_list = data

                #inputing the clean and filtered data into the list
                rejected_gender =[]

                for name in name_list:
                    cleaned_name = self.clean_name(name= name) #this will clean the name

                    if self.is_valid_name(cleaned_name): #this will only append the right valid names
                        final_list.append(cleaned_name)
                    else:
                        if return_value != 3:
                            print(f"\nWarning: '{name}' is not a valid Gender and was skipped\n")
                        
                        else:
                            rejected_gender.append(name)
                
                if return_value != 3:
                    if rejected_gender:
                        print(f"\n!!!\tWarning below name(s) are rejeted\t!!!\n")
                        print(rejected_gender)
                

                #check if the final final_list is empty or not
                if not final_list:
                    print("Error: No valid Genders were provided!\n")
                    self.counter()
                    
                    return getting_gender(choice=return_value)

                
                if return_value != 3:
                    condition = (self.data['name'].notna().sum() == len(final_list))
                else:
                    condition = (data.notna().sum() == len(final_list))

                if condition:

                    final_list = self.gender_conversion(final_list,True,False) #since the data is binary
                    print(f"\nYour list has Gender:\tMale :{final_list.count(1)}\tFemale :{final_list.count(0)}\n")

                    #converting to dataframes
                    gender_column = pd.DataFrame({
                        "gender":final_list
                    })

                else:
                    print(f"Error: Number of Gender Value does not match the number of input Names !\n")
                    self.counter
                    return getting_gender(choice=return_value)
                                   
                    
                return gender_column
            
            #get the geneder column for the data set 
            def getting_gender(data=self.data, choice=self.choice):

                if choice == 1:
                    gender_column = input("\nEnter 'Gender' (Male/Female):\t")
                    gender_column = gender_cleaned_list(data=self.data,gender_column=gender_column, return_value = 1)
                    df = gender_column

                elif choice == 2:
                    gender_column = input("\nEnter list with 'Genders' (Male/Female):\t")
                    gender_column = gender_cleaned_list(data=self.data,gender_column=gender_column, return_value = 2)
                    df = gender_column
                                       
                elif self.choice == 3:

                    #create a copy of self.data 
                    data = self.sheet
                    df = data.copy()

                    gender_column = input("\nEnter the exact column name with 'Genders':\t")
                    gender_column = gender_column.strip().strip('"') #this will help in removing space 
                    
                   
                    if gender_column not in df.columns:
                        #The only case of failure is at choice == 3 
                        print(f"'\n{gender_column}' is not available in the sheet\n")

                        #give one more option to enter the column name
                        self.counter()
                        return getting_gender(choice=3) #this will redirect into same function
                            
                    else:
                        if df['name'].count() == df[gender_column].count():

                            #does this contain a number assuming binary 
                            if not set([0, 1]).isdisjoint(df[gender_column].values):
                                
                                df[gender_column] = self.gender_conversion(
                                    Gender_Data=df[gender_column],
                                    GenderToNumber=False,
                                    NumberToGender=True
                                )
                            
                            #load the data otherwise
                            df = gender_cleaned_list(data=df[gender_column],gender_column=gender_column,return_value=3)
                        else:
                            print(f"Error: Gender Value of [{df[gender_column].count()}] does not match the input size of [{self.data['name'].count}] !\n")
                            #give one more option to enter the column name
                            self.counter()
                            return getting_gender(choice=3) #this will redirect into same function

                return df
                    
                
            expected_output  = getting_gender() #this will load the exact data from the dataframe
            

            if expected_output is not None and X_test == True :
                expected_output = pd.read_csv("model.help/Y_test.csv")
                print(f"\n !!!\t Input feature is None so using internal test set for prediction\t!!!\n")

            # converting the expected values to a list because prediction is a list
            expected_output = expected_output.values.tolist()
            prediction = np.array(self.prediction)
            
            count = 0
            for i in range(0,len(expected_output)):
                if expected_output[i] == prediction[i]:
                    count = count +1
            print(f"\n\n Accuracy of output is:\t{(count*100)/len(expected_output)}\n")

        else :
            #prompt to loop actions if needed 
            print("\nThank you\n")
            return None
        
       
    def gender_conversion(self,Gender_Data,GenderToNumber,NumberToGender):
        
        #converts the gender into binary 
        final_list=[]

        if GenderToNumber is True:
            for gender in Gender_Data:
                
                if str(gender).lower() == "male":
                    gender = 1
                    final_list.append(gender)
                
                elif str(gender).lower() == "female":
                    gender = 0
                    final_list.append(gender)
                else:
                    gender = "Issue"
                    final_list.append(gender)

        #converts the  binary into gender
        if NumberToGender is True:
            for gender in Gender_Data:
            
                if int(gender)== 1:
                    gender = str("Male")
                    final_list.append(gender)
                
                elif int(gender) == 0:
                    gender = str("Female")
                    final_list.append(gender)
                else:
                    gender = str("Issue")
                    final_list.append(gender)

        return final_list  


    def output_printing(self,predict_input,feature_name):
        pass
        
        

def main():
    try:
        Detector = GenderDetection()
        Detector.get_data()

    except Exception as e:
        print(f"\n!!!\tThere was an error : {e}\t!!!\n")        


if __name__ == "__main__":
    main()