from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd 
import numpy as np 
from pathlib import Path
import joblib 
import os
import re
import sys

#Generalising the file names for future use
MODEL_FILE = "model.help/model.pkl" 
PIPELINE_FILE = "model.help/pipeline.pkl"

#creating the modle.help for better handeling 
Path("model.help").mkdir(parents=True,exist_ok=True)

class model_training:
    def __init__(self):
        self.data = None
        self.file_name = "Preposed_model_training_data_of_name_gender_dataset.xlsx.csv" 
        #self.file_name = "Indian_firstname_Gender_Data.csv"
    

    def training(self):
        
        try:
            if not os.path.exists(MODEL_FILE):

                print("\n\nTraining the model ...........\n\n Please wait for a while!")
                #getting the data 
                df = self.data_input()

                #get the features and labels accrodingly 
                labels = pd.DataFrame(df['Gender'].copy())
                features = self.extract_name_features(df=df, name_col= "Name")

                #combine the data to make it the new dataFrame
                df = pd.concat([features, labels], axis = 1)
                
                '''Df is changed at this point'''

                #stratified split 
                split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

                for train_index,test_index in split.split(df,df["Gender"]):
                    df_train_set = pd.DataFrame(df.loc[train_index])
                    df_test_set = pd.DataFrame(df.loc[test_index])

                #saves the default test set to modle.help folder
                model_test_labels = df_test_set["Gender"].copy().to_csv("model.help/Y_test.csv", index = False)
                model_test_features = df_test_set.drop("Gender", axis =1).to_csv("model.help/X_test.csv", index = False)

                #extracting the label and features from the train set
                ''' This is very important as df_train is a new dataset'''

                model_labels = df_train_set["Gender"].copy()
                model_features = df_train_set.drop("Gender", axis =1)


                #Spliting cat_attributes and num_attributes in the form of list

                '''Need to change this if you change the features'''

                cat_attrib = ["first_letter","last_letter"]
                # cat_attrib = ["first_letter","last_letter","suffix_2","prefix_2"]
                num_attrib = model_features.drop(cat_attrib + ["name"], axis =1).columns.tolist()
                
                #putting into the pipeline for transformation
                pipeline = self.build_pipeline(num_attrib,cat_attrib)
                model_prepared = pipeline.fit_transform(model_features)

                #Traing the model
                model = RandomForestClassifier()
                model.fit(model_prepared,model_labels)

                joblib.dump(model,MODEL_FILE)
                joblib.dump(pipeline,PIPELINE_FILE)

                print("\n\t\t .... Model Training completed succefully....\n")


            else:

                if __name__ == "__main__":
                    #INFERENCE PHASE 
                    print("\nRunning model for prediction ......\n")
                    model = joblib.load(MODEL_FILE)
                    pipeline = joblib.load(PIPELINE_FILE)
                    
                    #now we need to get the data for prediction
                    '''Connect the user interaction and get data from there for prediction'''

                    
                    df = pd.read_csv("model.help/X_test.csv")
                    transformed_features = pipeline.transform(df)
                    prediction = model.predict(transformed_features)
                    df["prediction"] = prediction

                    df.to_csv("output.csv", index = False)
                    print("\n\t\t .... Inference completed succefully....\n")
                    print("Results saved to 'output.csv'")
                else:
                    print("\nModel has been trained already and Prediction initiated.......\n")
        
        except Exception as e:
            
            print(f"\n!!!\tThere was an error : {e}\t!!!\n")


    def data_input(self):
        try:
            
            '''INPUT TAKEN FROM THE __init__ module'''

            file_name = self.file_name

            '''Make sure you move it into "Data_file" folder'''
            final_path = Path.cwd()/"Data_files"/file_name  

            if not final_path.exists() :

                for i in range(2):
                    print("\n!!!\tUnable to locate the file\t!!!\n")

                    #since path cannot handle the string directly 
                    path = Path(input(f"Enter exact path to the file {file_name} below:\n").strip('"'))
                    
                    if path.exists():
                        final_path = path
                        break 
                    elif i==1:
                        print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                        print("\t------Thank you ------\n")
            
            '''change this accoding to extention'''   

            extention = self.get_file_extension(final_path)

            if extention ==".csv":
                df = pd.read_csv(final_path)
                
            elif extention ==".xlsx":
                df = pd.read_excel(final_path)
            else:
                print(f"\n!!!\t[{final_path}] is not accessible\n")

            '''For global use'''
            self.data = df 
            
            df = self.clean_input(data=df) #doing the data cleaning and preprocessing

        except Exception as e:
            print(f"\n!!!\tThere was an error : {e}\t!!!\n")
        
        return df
    
    #getting the extention
    def get_file_extension(self,filename):

        name,extention = os.path.splitext(filename)
        return extention.lower()
    
    #doing data cleaning and processing
    def clean_input(self,data):

        df = data.copy() #safer side 
        df = df[["Name","Gender"]] #only pass important info and remove the rest

        #separate them for easy processing
        gender =df["Gender"]
        name = df["Name"]
        
        processed_results = [] 
        rejected_results =[] 

        for i in range(len(name)):
            current_name = name.loc[i]
            current_gender = gender.iloc[i]
            
            '''Name preprosing'''

            current_name = self.clean_name(current_name)
            
            if self.is_valid_name(current_name): #this will only append the right valid names
                name_value = True
            else:
                name_value = False
                        
            '''Gender preprosing'''

            if current_gender == 1 or current_gender ==0 :
                gender_value = True
            else:
                gender_value = False

            #create a pair of this data

            processed_pair ={
                "Name": current_name,
                "Gender": current_gender
            }
            
            if name_value == True and gender_value ==True:
                processed_results.append(processed_pair)
            else:
                rejected_results.append(processed_pair)

        if rejected_results:
            print(f"\n!!!\tWarning below results(s) are rejeted\t!!!\n")
            print(pd.DataFrame(rejected_results))

        if not processed_results:
            print("Error: No valid names were provided to the model!\n")
            sys.exit(1)
        
        df = pd.DataFrame(processed_results)

        #handle datatype 
        if type(df['Name']) != "object":
            df["Name"] = df["Name"].astype("str")
        if type(df['Gender']).dtype != "int64":
            df["Gender"] = df["Gender"].astype("int64")
        
        
        output_file = str(f"Data_files/Preposed_model_training_data_of_{self.file_name}.csv")

        if not os.path.exists(output_file): 
            df.to_csv(output_file, index= False)
            self.data = df
        
        return df

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


    '''These features are connected to user_interaction.py file'''

    #getting the features for the data 
    def extract_name_features(self,df, name_col ='name'):
        df['name'] = df[name_col].str.strip().str.lower()
        df["name_length"] = df["name"].apply(len)
        df["first_letter"] = df["name"].str[0]
        df["last_letter"] = df["name"].str[-1]
        df["vowel_count"] = df["name"].apply(lambda x: sum(1 for c in x if c in "aeiou"))
        df["consonant_count"] = df["name_length"] - df["vowel_count"]
        # df["suffix_2"] = df["name"].str[-2:]
        # df["prefix_2"] = df["name"].str[:2]
        df["is_last_letter_a"] = (df["last_letter"]== "a").astype(int)

        #rearranging the df accordingly 
        df_function = df[["name","name_length","first_letter","last_letter","vowel_count","consonant_count","is_last_letter_a"]].copy()
        # df_function = df[["name","name_length","first_letter","last_letter","vowel_count",	"consonant_count",	"suffix_2","prefix_2","is_last_letter_a"]].copy()
        
        return df_function

    #Building the complete pipeline for data transfromation
    def build_pipeline(self,num_attributes,cat_attributes):

        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder())
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline,num_attributes),
            ("cat", cat_pipeline, cat_attributes)
        ])
    
        return full_pipeline

def main():
    model = model_training()
    model.training()

#running the program
if __name__ == "__main__":
    main()

