import pandas as pd 
import numpy as np 
import os
from pathlib import Path
import joblib 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Generalising the file names for future use
MODEL_FILE = "model.help/model.pkl" 
PIPELINE_FILE = "model.help/pipeline.pkl"

#creating the modle.help for better handeling 
Path("model.help").mkdir(parents=True,exist_ok=True)

#getting the features for the data 
def extract_name_features(df, name_col ='name'):
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

#Building the complete pipeline for data transfromation
def build_pipeline(num_attributes,cat_attributes):

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

def data_input():
    try:
        #handle FileNotFoundError
        final_path = Path.cwd()/"Data_files"/"Indian_firstname_Gender_Data.csv"  

        if not final_path.exists() :

            for i in range(2):
                print("\n!!!\tUnable to locate the file\t!!!\n")

                #since path cannot handle the string directly 
                path = Path(input("Enter exact path to the file 'Indian_firstname_Gender_Data.csv' below:\n").strip('"'))
                
                if path.exists():
                    final_path = path
                    break 
                elif i==1:
                    print("\n!!!\tYou have excceded the repeat limit, Try again later\t!!!\n")
                    print("\t------Thank you ------\n")
                
        #this is the "Indian_firstname_Gender_Data.csv" 
        df = pd.read_csv(final_path)
    
    except Exception as e:
        print(f"\n!!!\tThere was an error : {e}\t!!!\n")
    
    return df


if not os.path.exists(MODEL_FILE):
    try:

        print("\n\nTraining the model ...........\n\n Please wait for a while!")
        #getting the data 
        df = data_input()

        #get the features and labels accrodingly 
        labels = pd.DataFrame(df['Gender'].copy())
        features = extract_name_features(df=df, name_col= "Name")

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

        cat_attrib = ["first_letter","last_letter","suffix_2","prefix_2"]
        num_attrib = model_features.drop(cat_attrib + ["name"], axis =1).columns.tolist()
        
        #putting into the pipeline for transformation
        pipeline = build_pipeline(num_attrib,cat_attrib)
        model_prepared = pipeline.fit_transform(model_features)

        #Traing the model
        model = RandomForestClassifier()
        model.fit(model_prepared,model_labels)

        joblib.dump(model,MODEL_FILE)
        joblib.dump(pipeline,PIPELINE_FILE)

        print("\n\t\t .... Model Training completed succefully....\n")

    except Exception as e:
        
        print(f"\n!!!\tThere was an error : {e}\t!!!\n")

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


