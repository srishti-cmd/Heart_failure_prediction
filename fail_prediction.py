"""
Heart Failure Readmission Prediction Model
-----------------------------------------
This code builds a machine learning model to predict 30-day readmission for heart failure patients
using the MIMIC-III dataset.

Components:
- Data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Model development and evaluation
- Results visualization

Author: [Your Name]
Date: May 10, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# List of heart failure ICD-9 codes (from problem statement)
HEART_FAILURE_ICD9_CODES = [
    '39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', 
    '40493', '4280', '4281', '42820', '42821', '42822', '42823', '42830', '42831', 
    '42832', '42833', '42840', '42841', '42842', '42843', '4289'
]

class HeartFailureReadmissionPredictor:
    """
    A class to predict 30-day readmission risk for heart failure patients
    """
    
    def __init__(self, data_path):
        """
        Initialize the predictor with the path to the data
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing MIMIC-III data files
        """
        self.data_path = data_path
        self.admissions_df = None
        self.patients_df = None
        self.diagnoses_df = None
        self.procedures_df = None
        self.prescriptions_df = None
        self.lab_results_df = None
        self.heart_failure_patients = None
        self.final_dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """
        Load necessary tables from MIMIC-III
        """
        print("Loading data files...")
        
        # Load admissions data
        self.admissions_df = pd.read_csv(os.path.join(self.data_path, 'ADMISSIONS.csv'))
        print(f"Loaded admissions data: {self.admissions_df.shape}")
        
        # Load patients data
        self.patients_df = pd.read_csv(os.path.join(self.data_path, 'PATIENTS.csv'))
        print(f"Loaded patients data: {self.patients_df.shape}")
        
        # Load diagnoses data
        self.diagnoses_df = pd.read_csv(os.path.join(self.data_path, 'DIAGNOSES_ICD.csv'))
        print(f"Loaded diagnoses data: {self.diagnoses_df.shape}")
        
        # Load procedures data (if available)
        try:
            self.procedures_df = pd.read_csv(os.path.join(self.data_path, 'PROCEDURES_ICD.csv'))
            print(f"Loaded procedures data: {self.procedures_df.shape}")
        except FileNotFoundError:
            print("Procedures data not found. Continuing without it.")
        
        # Load prescriptions data (if available)
        try:
            self.prescriptions_df = pd.read_csv(os.path.join(self.data_path, 'PRESCRIPTIONS.csv'))
            print(f"Loaded prescriptions data: {self.prescriptions_df.shape}")
        except FileNotFoundError:
            print("Prescriptions data not found. Continuing without it.")
        
        # Load lab results data (if available)
        try:
            self.lab_results_df = pd.read_csv(os.path.join(self.data_path, 'LABEVENTS.csv'))
            print(f"Loaded lab results data: {self.lab_results_df.shape}")
        except FileNotFoundError:
            try:
                self.lab_results_df = pd.read_csv(os.path.join(self.data_path, 'LAB_EVENTS.csv'))
                print(f"Loaded lab results data: {self.lab_results_df.shape}")
            except FileNotFoundError:
                print("Lab events data not found. Continuing without it.")
                
        print("Data loading complete!")
        
    def _convert_to_datetime(self, df, date_columns):
        """
        Convert date columns to datetime format
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing date columns
        date_columns : list
            List of column names to convert
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with converted date columns
        """
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
        
    def identify_heart_failure_patients(self):
        """
        Identify patients with heart failure diagnoses
        """
        print("Identifying heart failure patients...")
        
        # Filter diagnoses to include only heart failure ICD-9 codes
        heart_failure_diagnoses = self.diagnoses_df[
            self.diagnoses_df['ICD9_CODE'].isin(HEART_FAILURE_ICD9_CODES)
        ]
        
        # Get unique patient IDs with heart failure
        heart_failure_patient_ids = heart_failure_diagnoses['SUBJECT_ID'].unique()
        print(f"Found {len(heart_failure_patient_ids)} patients with heart failure diagnoses")
        
        # Get all admissions for these patients
        self.heart_failure_patients = self.admissions_df[
            self.admissions_df['SUBJECT_ID'].isin(heart_failure_patient_ids)
        ].copy()
        
        # Convert date columns to datetime
        self.heart_failure_patients = self._convert_to_datetime(
            self.heart_failure_patients, 
            ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME']
        )
        
        # Sort by patient ID and admission time
        self.heart_failure_patients.sort_values(['SUBJECT_ID', 'ADMITTIME'], inplace=True)
        
        print(f"Processing {self.heart_failure_patients.shape[0]} admissions for heart failure patients")
        
    def create_readmission_labels(self):
        """
        Create 30-day readmission labels for each admission
        """
        print("Creating readmission labels...")
        
        # Calculate time difference between consecutive admissions for the same patient
        self.heart_failure_patients['NEXT_ADMITTIME'] = self.heart_failure_patients.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
        
        # Calculate days until next admission
        self.heart_failure_patients['DAYS_NEXT_ADMIT'] = (
            self.heart_failure_patients['NEXT_ADMITTIME'] - self.heart_failure_patients['DISCHTIME']
        ).dt.total_seconds() / (24 * 60 * 60)
        
        # Create the target variable: 1 if readmitted within 30 days, 0 otherwise
        self.heart_failure_patients['READMISSION_30D'] = (
            (self.heart_failure_patients['DAYS_NEXT_ADMIT'] >= 0) & 
            (self.heart_failure_patients['DAYS_NEXT_ADMIT'] <= 30)
        ).astype(int)
        
        # Remove rows where we can't determine readmission status (last admission for each patient)
        self.heart_failure_patients = self.heart_failure_patients.dropna(subset=['DAYS_NEXT_ADMIT'])
        
        print(f"Created labels for {self.heart_failure_patients.shape[0]} admissions")
        print(f"Readmission rate: {self.heart_failure_patients['READMISSION_30D'].mean():.2%}")
        
    def extract_demographics(self):
        """
        Extract and add demographic features from the PATIENTS table
        """
        print("Extracting demographic features...")
        
        # Merge patient demographics
        demographics = self.patients_df[['SUBJECT_ID', 'GENDER', 'DOB']]
        self.heart_failure_patients = self.heart_failure_patients.merge(
            demographics, on='SUBJECT_ID', how='left'
        )
        
        # Calculate age at admission
        self.heart_failure_patients['DOB'] = pd.to_datetime(self.heart_failure_patients['DOB'], errors='coerce')
        self.heart_failure_patients['AGE'] = (
            self.heart_failure_patients['ADMITTIME'].dt.year - 
            self.heart_failure_patients['DOB'].dt.year
        )
        
        # Cap age at 90 for privacy (as often done with MIMIC)
        self.heart_failure_patients.loc[self.heart_failure_patients['AGE'] > 90, 'AGE'] = 90
        
        # Create age groups
        self.heart_failure_patients['AGE_GROUP'] = pd.cut(
            self.heart_failure_patients['AGE'], 
            bins=[0, 30, 50, 70, 90, 200],
            labels=['0-30', '31-50', '51-70', '71-90', '90+']
        )
        
        print("Demographics extracted successfully")
        
    def extract_admission_features(self):
        """
        Extract features related to hospital admission
        """
        print("Extracting admission-related features...")
        
        # Length of stay (days)
        self.heart_failure_patients['LOS'] = (
            self.heart_failure_patients['DISCHTIME'] - 
            self.heart_failure_patients['ADMITTIME']
        ).dt.total_seconds() / (24 * 60 * 60)
        
        # Cap extreme values of length of stay
        upper_los = self.heart_failure_patients['LOS'].quantile(0.99)
        self.heart_failure_patients.loc[self.heart_failure_patients['LOS'] > upper_los, 'LOS'] = upper_los
        
        # Emergency admission flag
        self.heart_failure_patients['IS_EMERGENCY'] = (
            self.heart_failure_patients['ADMISSION_TYPE'] == 'EMERGENCY'
        ).astype(int)
        
        # Weekend admission flag
        self.heart_failure_patients['IS_WEEKEND'] = (
            self.heart_failure_patients['ADMITTIME'].dt.dayofweek >= 5
        ).astype(int)
        
        # Previous admissions count
        admission_counts = self.heart_failure_patients.groupby('SUBJECT_ID').cumcount()
        self.heart_failure_patients['PREV_ADMISSIONS'] = admission_counts
        
        # Insurance type
        self.heart_failure_patients['INSURANCE_MEDICARE'] = (
            self.heart_failure_patients['INSURANCE'] == 'Medicare'
        ).astype(int)
        
        self.heart_failure_patients['INSURANCE_MEDICAID'] = (
            self.heart_failure_patients['INSURANCE'] == 'Medicaid'
        ).astype(int)
        
        self.heart_failure_patients['INSURANCE_PRIVATE'] = (
            self.heart_failure_patients['INSURANCE'] == 'Private'
        ).astype(int)
        
        print("Admission features extracted successfully")
        
    def extract_comorbidities(self):
        """
        Extract comorbidity information for each admission
        """
        print("Extracting comorbidity features...")
        
        # Define common comorbidities with their ICD-9 codes
        comorbidities = {
            'HYPERTENSION': ['401', '402', '403', '404', '405'],
            'DIABETES': ['250'],
            'COPD': ['490', '491', '492', '493', '494', '495', '496'],
            'KIDNEY_DISEASE': ['585', '586', '403', '404'],
            'ATRIAL_FIBRILLATION': ['42731'],
            'ISCHEMIC_HEART': ['410', '411', '412', '413', '414'],
            'STROKE': ['430', '431', '432', '433', '434', '435', '436', '437', '438'],
            'ANEMIA': ['280', '281', '282', '283', '284', '285'],
            'DEPRESSION': ['296', '3004', '309', '311']
        }
        
        # Initialize comorbidity columns
        for comorbidity in comorbidities:
            self.heart_failure_patients[comorbidity] = 0
        
        # For each admission, check for presence of comorbidities
        for hadm_id in self.heart_failure_patients['HADM_ID'].unique():
            # Get diagnoses for this admission
            admission_diagnoses = self.diagnoses_df[self.diagnoses_df['HADM_ID'] == hadm_id]['ICD9_CODE'].tolist()
            
            # Check each comorbidity
            for comorbidity, codes in comorbidities.items():
                # Check if any diagnosis code starts with any of the comorbidity codes
                has_comorbidity = any(
                    any(str(diag).startswith(code) for code in codes) 
                    for diag in admission_diagnoses
                )
                
                if has_comorbidity:
                    self.heart_failure_patients.loc[
                        self.heart_failure_patients['HADM_ID'] == hadm_id, comorbidity
                    ] = 1
        
        # Calculate Charlson Comorbidity Index (simplified version)
        self.heart_failure_patients['COMORBIDITY_COUNT'] = self.heart_failure_patients[list(comorbidities.keys())].sum(axis=1)
        
        print("Comorbidity features extracted successfully")
        
    def extract_medication_features(self):
        """
        Extract medication-related features
        """
        print("Extracting medication features...")
        
        if self.prescriptions_df is None:
            print("Prescriptions data not available. Skipping medication features.")
            return
        
        # Define common heart failure medications
        hf_medications = {
            'ACE_INHIBITOR': ['LISINOPRIL', 'ENALAPRIL', 'CAPTOPRIL', 'RAMIPRIL', 'QUINAPRIL', 'BENAZEPRIL', 'FOSINOPRIL'],
            'ARB': ['LOSARTAN', 'VALSARTAN', 'CANDESARTAN', 'IRBESARTAN', 'OLMESARTAN', 'TELMISARTAN'],
            'BETA_BLOCKER': ['METOPROLOL', 'CARVEDILOL', 'BISOPROLOL', 'ATENOLOL', 'PROPRANOLOL', 'LABETALOL'],
            'DIURETIC': ['FUROSEMIDE', 'HYDROCHLOROTHIAZIDE', 'BUMETANIDE', 'TORSEMIDE', 'SPIRONOLACTONE', 'CHLORTHALIDONE'],
            'DIGOXIN': ['DIGOXIN'],
            'ANTICOAGULANT': ['WARFARIN', 'HEPARIN', 'ENOXAPARIN', 'DABIGATRAN', 'RIVAROXABAN', 'APIXABAN', 'EDOXABAN'],
            'STATIN': ['ATORVASTATIN', 'SIMVASTATIN', 'ROSUVASTATIN', 'PRAVASTATIN', 'LOVASTATIN', 'FLUVASTATIN', 'PITAVASTATIN']
        }
        
        # Initialize medication columns
        for med_class in hf_medications:
            self.heart_failure_patients[med_class] = 0
        
        # Count of total unique medications
        med_counts = self.prescriptions_df.groupby('HADM_ID')['DRUG'].nunique().reset_index()
        med_counts.columns = ['HADM_ID', 'UNIQUE_MED_COUNT']
        
        # Merge medication counts into patient data
        self.heart_failure_patients = self.heart_failure_patients.merge(
            med_counts, on='HADM_ID', how='left'
        )
        self.heart_failure_patients['UNIQUE_MED_COUNT'].fillna(0, inplace=True)
        
        # For each admission, check for presence of key medications
        for hadm_id in self.heart_failure_patients['HADM_ID'].unique():
            # Get medications for this admission
            admission_meds = self.prescriptions_df[self.prescriptions_df['HADM_ID'] == hadm_id]['DRUG'].str.upper().tolist()
            
            # Check each medication class
            for med_class, drugs in hf_medications.items():
                # Check if any medication in this class was prescribed
                has_med = any(
                    any(drug in med.upper() for drug in drugs) 
                    for med in admission_meds if isinstance(med, str)
                )
                
                if has_med:
                    self.heart_failure_patients.loc[
                        self.heart_failure_patients['HADM_ID'] == hadm_id, med_class
                    ] = 1
        
        print("Medication features extracted successfully")
        
    def extract_lab_features(self):
        """
        Extract laboratory test results
        """
        print("Extracting laboratory features...")
        
        if self.lab_results_df is None:
            print("Lab results data not available. Skipping lab features.")
            return
        
        # Define important lab tests for heart failure
        important_labs = {
            'SODIUM': [50983],           # Sodium
            'POTASSIUM': [50971],        # Potassium
            'BUN': [51006],              # Blood Urea Nitrogen
            'CREATININE': [50912],       # Creatinine
            'GLUCOSE': [50931],          # Glucose
            'HGB': [50811],              # Hemoglobin
            'WBC': [51301],              # White Blood Cell Count
            'PLATELETS': [51265],        # Platelets
            'BNP': [50963],              # Brain Natriuretic Peptide
            'TROPONIN': [51002, 51003]   # Troponin I and T
        }
        
        # For each lab test, get the first value during each admission
        for lab_name, item_ids in important_labs.items():
            # Filter lab events to this test and merge with admissions
            lab_data = self.lab_results_df[self.lab_results_df['ITEMID'].isin(item_ids)].copy()
            
            if len(lab_data) == 0:
                continue
                
            # Convert to numeric, ignoring errors
            lab_data['VALUENUM'] = pd.to_numeric(lab_data['VALUENUM'], errors='coerce')
            
            # Get the first value for each admission
            first_labs = lab_data.sort_values('CHARTTIME').groupby('HADM_ID')['VALUENUM'].first().reset_index()
            first_labs.columns = ['HADM_ID', lab_name]
            
            # Merge into patient data
            self.heart_failure_patients = self.heart_failure_patients.merge(
                first_labs, on='HADM_ID', how='left'
            )
        
        print("Laboratory features extracted successfully")
        
    def prepare_final_dataset(self):
        """
        Prepare the final dataset for modeling
        """
        print("Preparing final dataset...")
        
        # Select relevant features
        features = ['SUBJECT_ID', 'HADM_ID', 'READMISSION_30D', 'AGE', 'GENDER', 'LOS', 
                'IS_EMERGENCY', 'IS_WEEKEND', 'PREV_ADMISSIONS', 'INSURANCE_MEDICARE', 
                'INSURANCE_MEDICAID', 'INSURANCE_PRIVATE', 'COMORBIDITY_COUNT']
        
        # Add comorbidity features
        comorbidity_features = ['HYPERTENSION', 'DIABETES', 'COPD', 'KIDNEY_DISEASE', 
                            'ATRIAL_FIBRILLATION', 'ISCHEMIC_HEART', 'STROKE', 
                            'ANEMIA', 'DEPRESSION']
        features.extend([f for f in comorbidity_features if f in self.heart_failure_patients.columns])
        
        # Add medication features
        med_features = ['ACE_INHIBITOR', 'ARB', 'BETA_BLOCKER', 'DIURETIC', 'DIGOXIN', 
                    'ANTICOAGULANT', 'STATIN', 'UNIQUE_MED_COUNT']
        features.extend([f for f in med_features if f in self.heart_failure_patients.columns])
        
        # Add lab features
        lab_features = ['SODIUM', 'POTASSIUM', 'BUN', 'CREATININE', 'GLUCOSE', 'HGB', 
                    'WBC', 'PLATELETS', 'BNP', 'TROPONIN']
        features.extend([f for f in lab_features if f in self.heart_failure_patients.columns])
        
        # Keep only selected features
        self.final_dataset = self.heart_failure_patients[
            [f for f in features if f in self.heart_failure_patients.columns]
        ].copy()
        
        # Drop rows with too many missing values (>50%)
        self.final_dataset = self.final_dataset.dropna(thresh=len(self.final_dataset.columns)//2)
        
        print(f"Final dataset shape: {self.final_dataset.shape}")
        print(f"Features included: {len(self.final_dataset.columns) - 3}")  # Excluding ID columns and target
        print(f"Final readmission rate: {self.final_dataset['READMISSION_30D'].mean():.2%}")
    
    def perform_eda(self, output_dir='outputs'):
        """
        Perform exploratory data analysis and save visualizations
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        print("Performing exploratory data analysis...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribution of readmission
        plt.figure(figsize=(8, 6))
        sns.countplot(x='READMISSION_30D', data=self.final_dataset)
        plt.title('Distribution of 30-Day Readmissions')
        plt.xlabel('Readmitted within 30 days')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, '01_readmission_distribution.png'))
        plt.close()
        
        # 2. Age distribution by readmission status
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.final_dataset, x='AGE', hue='READMISSION_30D', 
                    multiple='dodge', bins=10)
        plt.title('Age Distribution by Readmission Status')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, '02_age_distribution.png'))
        plt.close()
        
        # 3. Length of stay by readmission status
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='READMISSION_30D', y='LOS', data=self.final_dataset)
        plt.title('Length of Stay by Readmission Status')
        plt.xlabel('Readmitted within 30 days')
        plt.ylabel('Length of Stay (days)')
        plt.savefig(os.path.join(output_dir, '03_los_by_readmission.png'))
        plt.close()
        
        # 4. Comorbidity count by readmission status
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='READMISSION_30D', y='COMORBIDITY_COUNT', data=self.final_dataset)
        plt.title('Comorbidity Count by Readmission Status')
        plt.xlabel('Readmitted within 30 days')
        plt.ylabel('Number of Comorbidities')
        plt.savefig(os.path.join(output_dir, '04_comorbidities_by_readmission.png'))
        plt.close()
        
        # 5. Readmission rate by comorbidity
        comorbidity_features = [col for col in self.final_dataset.columns if col in 
                            ['HYPERTENSION', 'DIABETES', 'COPD', 'KIDNEY_DISEASE', 
                                'ATRIAL_FIBRILLATION', 'ISCHEMIC_HEART', 'STROKE', 
                                'ANEMIA', 'DEPRESSION']]
        
        if comorbidity_features:
            readmit_by_comorbidity = {}
            for comorbidity in comorbidity_features:
                if comorbidity in self.final_dataset.columns:
                    has_condition = self.final_dataset[self.final_dataset[comorbidity] == 1]['READMISSION_30D'].mean()
                    no_condition = self.final_dataset[self.final_dataset[comorbidity] == 0]['READMISSION_30D'].mean()
                    readmit_by_comorbidity[comorbidity] = [has_condition, no_condition]
            
            if readmit_by_comorbidity:
                plt.figure(figsize=(12, 8))
                data = pd.DataFrame(readmit_by_comorbidity, index=['Has Condition', 'No Condition']).T
                data.plot(kind='bar', figsize=(12, 8))
                plt.title('Readmission Rate by Comorbidity')
                plt.xlabel('Comorbidity')
                plt.ylabel('Readmission Rate')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '05_readmission_by_comorbidity.png'))
                plt.close()
        
        # 6. Correlation matrix
        numeric_cols = self.final_dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) > 2:  # Need at least a few numeric columns
            plt.figure(figsize=(14, 12))
            corr_matrix = self.final_dataset[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, linewidths=.5)
            plt.title('Correlation Matrix of Numeric Features')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '06_correlation_matrix.png'))
            plt.close()
            
        # 7. Create a summary statistics table
        summary_stats = self.final_dataset.describe().T
        summary_stats['missing'] = self.final_dataset.isnull().sum()
        summary_stats['missing_pct'] = self.final_dataset.isnull().sum() / len(self.final_dataset)
        summary_stats.to_csv(os.path.join(output_dir, '07_summary_statistics.csv'))
        
        print("EDA completed and saved to output directory")
    
    def prepare_train_test_data(self):
        """
        Prepare data for modeling by splitting into train and test sets
        """
        print("Preparing training and testing datasets...")
        
        # Separate features and target
        X = self.final_dataset.drop(['SUBJECT_ID', 'HADM_ID', 'READMISSION_30D'], axis=1, errors='ignore')
        y = self.final_dataset['READMISSION_30D']
        
        # Save feature names for later interpretation
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training data: {self.X_train.shape}")
        print(f"Testing data: {self.X_test.shape}")
        
    def build_preprocessing_pipeline(self):
        """
        Build a preprocessing pipeline for numerical and categorical features
        
        Returns:
        --------
        sklearn.compose.ColumnTransformer
            Preprocessing pipeline
        """
        # Identify numerical and categorical columns
        numerical_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Preprocessing for numerical data: impute missing values and scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical data: impute missing values and one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'  # Drop columns that aren't specified
        )
        
        return preprocessor
    
    def train_and_evaluate_models(self):
        """
        Train multiple models and evaluate their performance
        """
        print("Training and evaluating models...")
        
        # Build preprocessing pipeline
        preprocessor = self.build_preprocessing_pipeline()
        
        # Define models to evaluate
        models = {
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=sum(self.y_train == 0)/sum(self.y_train == 1))
        }
        
        # Dictionary to store results
        results = {}
        
        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Cross-validation performance
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
            
            print(f"{name} CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Train model on full training set
            pipeline.fit(self.X_train, self.y_train)
            
            # Predict on test set
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = average_precision_score(self.y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'pipeline': pipeline,
                'cv_scores': cv_scores,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'roc_curve': (fpr, tpr, roc_auc),
                'pr_curve': (precision, recall, pr_auc),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            # Print test performance
            print(f"{name} Test ROC-AUC: {roc_auc:.4f}")
            print(f"{name} Test PR-AUC: {pr_auc:.4f}")
            print(f"Classification Report:\n{classification_report(self.y_test, y_pred)}")
        
        # Determine best model based on test ROC-AUC
        best_model_name = max(results, key=lambda x: results[x]['roc_curve'][2])
        self.best_model = results[best_model_name]['pipeline']
        
        print(f"\nBest model: {best_model_name} with ROC-AUC: {results[best_model_name]['roc_curve'][2]:.4f}")
        
        return results
    
    def plot_model_performance(self, results, output_dir='outputs'):
        """
        Plot performance metrics for all models
        
        Parameters:
        -----------
        results : dict
            Dictionary with model results
        output_dir : str
            Directory to save output files
        """
        print("Plotting model performance...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. ROC curves for all models
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, roc_auc = result['roc_curve']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, '08_roc_curves.png'))
        plt.close()
        
        # 2. Precision-Recall curves for all models
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            precision, recall, pr_auc = result['pr_curve']
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, '09_precision_recall_curves.png'))
        plt.close()
        
        # 3. Confusion matrix for best model
        best_model_name = max(results, key=lambda x: results[x]['roc_curve'][2])
        cm = results[best_model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, '10_confusion_matrix.png'))
        plt.close()
        
        # 4. Feature importance for best model (if applicable)
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            model = results[best_model_name]['pipeline'].named_steps['model']
            
            # Get feature names after preprocessing (challenging due to transformations)
            # This is a simplified approach
            try:
                preprocessor = results[best_model_name]['pipeline'].named_steps['preprocessor']
                feature_names = []
                
                # Try to get numerical feature names (they stay the same)
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'num':
                            feature_names.extend(cols)
                
                # For categorical features, the names change due to one-hot encoding
                # This is a simplified approximation
                
                # Get feature importances
                importances = model.feature_importances_
                if len(importances) == len(feature_names):
                    # If the number of features match
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                else:
                    # Just use indices if we can't match names correctly
                    importance_df = pd.DataFrame({
                        'Feature': [f"Feature_{i}" for i in range(len(importances))],
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                
                # Plot top 15 features
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
                plt.title(f'Top 15 Feature Importances - {best_model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '11_feature_importance.png'))
                plt.close()
            except Exception as e:
                print(f"Could not plot feature importances: {e}")
        
        # 5. Cross-validation scores comparison
        cv_means = [results[name]['cv_scores'].mean() for name in results]
        cv_stds = [results[name]['cv_scores'].std() for name in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(results)), cv_means, yerr=cv_stds, 
            tick_label=list(results.keys()), alpha=0.7)
        plt.title('Cross-Validation ROC-AUC Scores')
        plt.ylabel('ROC-AUC Score')
        plt.ylim([min(cv_means) - 0.1, 1.0])
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '12_cv_scores.png'))
        plt.close()
        
        print("Performance plots saved to output directory")
    
    def save_model(self, output_dir='outputs'):
        """
        Save the best model and related artifacts
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        print("Saving model and artifacts...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        if self.best_model is not None:
            model_path = os.path.join(output_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Best model saved to {model_path}")
            
            # Save feature names
            feature_path = os.path.join(output_dir, 'feature_names.pkl')
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            print(f"Feature names saved to {feature_path}")
            
            # Create a simple README with instructions
            readme_path = os.path.join(output_dir, 'README.md')
            with open(readme_path, 'w') as f:
                f.write("# Heart Failure Readmission Prediction Model\n\n")
                f.write("## Overview\n")
                f.write("This model predicts the risk of 30-day readmission for heart failure patients.\n\n")
                f.write("## Files\n")
                f.write("- `best_model.pkl`: The trained prediction model\n")
                f.write("- `feature_names.pkl`: Names of features used by the model\n")
                f.write("- Several visualization files showing model performance\n\n")
                f.write("## Usage\n")
                f.write("```python\n")
                f.write("import pickle\n\n")
                f.write("# Load the model\n")
                f.write("with open('best_model.pkl', 'rb') as f:\n")
                f.write("    model = pickle.load(f)\n\n")
                f.write("# Load feature names\n")
                f.write("with open('feature_names.pkl', 'rb') as f:\n")
                f.write("    feature_names = pickle.load(f)\n\n")
                f.write("# Make predictions on new data\n")
                f.write("# Note: New data should have the same structure as training data\n")
                f.write("predictions = model.predict_proba(new_patient_data)[:, 1]\n")
                f.write("```\n")
            print(f"README file created at {readme_path}")
        else:
            print("No model to save. Please train models first.")
            
    def run_pipeline(self, data_path, output_dir='outputs'):
        """
        Run the complete pipeline from data loading to model evaluation
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing MIMIC-III data files
        output_dir : str
            Directory to save output files
        """
        print("Starting the pipeline...")
        
        # Data loading and preprocessing
        self.load_data()
        self.identify_heart_failure_patients()
        self.create_readmission_labels()
        self.extract_demographics()
        self.extract_admission_features()
        self.extract_comorbidities()
        self.extract_medication_features()
        self.extract_lab_features()
        self.prepare_final_dataset()
        
        # Exploratory data analysis
        self.perform_eda(output_dir)
        
        # Model training and evaluation
        self.prepare_train_test_data()
        results = self.train_and_evaluate_models()
        self.plot_model_performance(results, output_dir)
        self.save_model(output_dir)
        
        print("Pipeline completed successfully!")


def predict_readmission_risk(patient_data, model_path='outputs/best_model.pkl'):
    """
    Predicts readmission risk for a new patient
    
    Parameters:
    -----------
    patient_data : pandas.DataFrame
        Patient data with required features
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    float
        Probability of readmission within 30 days
    """
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make prediction
    risk_score = model.predict_proba(patient_data)[:, 1]
    
    return risk_score


if __name__ == "__main__":
    # Example usage
    data_path = "C:/Users/kirti/Prediction/heart-failure-predication/data/mimic-iii"  # Path to MIMIC-III data files
    output_dir = "C:/Users/kirti/Prediction/heart-failure-predication/outputs"  # Output directory
    
    # Create and run the pipeline
    predictor = HeartFailureReadmissionPredictor(data_path)
    predictor.run_pipeline(data_path, output_dir)