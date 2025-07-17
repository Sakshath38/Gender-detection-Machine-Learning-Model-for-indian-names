# Gender Detection Model for Indian Names
---

## System Architecture

The project is structured into two main script files:

- **`model.py`**: Handles all aspects of model building, feature engineering, training, and saving.
- **`user_interaction.py`**: Manages user inputs, data pre-processing, feature extraction, validation, prediction, and output generation.

---

### 1. User Interaction Flow

#### Input Flexibility

**Supported Input Types:**
- Single name (string)
- List of names (Python list or CSV/Excel columns)
- Excel/CSV sheet containing names

#### Processing Steps

1. **Input Collection and Validation**
   - Identifies input type: string, list, or file upload
   - Checks if file has valid `.csv` or `.xlsx` extension; throws error if invalid

2. **Data Pre-processing**
   - Extracts and standardizes first names
   - Cleans names via custom modules (removes spaces, special characters)
   - Validates name format and file structure
   - For file inputs, confirms presence of required columns and cleans as needed

3. **Feature Generation**
   - Imports feature logic from `model.py`
   - Ensures all required features are derived for modeling

4. **Model Invocation**
   - Passes pre-processed, feature-rich data to prediction pipeline
   - Stores predictions in designated output file

5. **Accuracy Checking (Optional)**
   - Prompts user to provide true gender labels if available
   - Validates gender column and converts to binary form if necessary
   - Compares predictions with ground truth, computes accuracy

---

### 2. Model File (`model.py`) Logic

#### Model Design

- **Algorithm**: Random Forest Classifier
- **Key Features Used:**
  - First Name
  - Last Name
  - Name Length
  - Whether name ends with a vowel (*corrected from "ends with 0"*)


#### Workflow

- **Feature Engineering**: Extracts relevant features from names
- **Preprocessing**: Converts names to consistent, clean strings; applies one-hot encoding and scaling as needed
- **Training**: Fits Random Forest on labeled data
- **Saving**: Persists trained model using `joblib` for efficient loading and reuse
- **Interoperability**: Allows `user_interaction.py` to import features and parameters directly

#### Model Resilience

Handles unseen or non-standard user data by:
- Prompting for correct file type or structure
- Ensuring only valid names and genders are processed
- Requesting binary gender conversion for non-binary inputs

---

### 3. Project Workflow (User Perspective)

1. **Submit Data**: Enter a single name, a list, or upload a file.
2. **Validation**: System ensures data format is supported and cleans entries.
3. **Feature Extraction**: Names transformed into model-ready features.
4. **Prediction**: Model predicts and outputs gender labels.
5. **Accuracy Check (Optional)**: If actual genders provided, computes and displays model accuracy.

---
