import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sklearn
import shap 
import time
import math
import sys 
import pathlib

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# must manually state categorical and target variables #

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

df_name = 'Supercapacitor.csv' # state csv file name
target_name = 'CAP' # state the target column name

has_categorical = False # example True or False (only for alphabet categorical values) 
ordinal_list = [] # example ['TEST','TEST3']
categories_list = [] # example [[Small,Medium,Large],[First,Second,Third]]
nominal_list = [] # example ['TEST','TEST3']

run = 'Both' # example 'Standalone','Stack','Both'

pathlib.Path(f'CSV').mkdir(parents=True,exist_ok=True)
pathlib.Path(f'CSV_models').mkdir(parents=True,exist_ok=True)
pathlib.Path(f'CSV_scale_target_original').mkdir(parents=True,exist_ok=True)

# SECTION Standalone Model #

class Process:

    def __init__(self,model=None,model_name=None,randomize=None,has_categorical=None):

        self.randomize_value_list = [x*0.01 for x in range(0,110,10)]
        self.model_name = model_name
        self.model = model
        self.has_categorical = has_categorical
        print(self.model_name)
        self.combined_df_list = []
        self.randomize = randomize
        self.process()

    def process(self):

        for random_seed in range(10):

            self.random_seed = random_seed
            self.random_state = np.random.RandomState(self.random_seed)
            self.combined_df = pd.DataFrame(columns=[f'NOISEPERCENTSEED{self.random_seed}',f'AVGSEED{self.random_seed}',f'TIMESEED{self.random_seed}'])
                
            print(f'random seed {random_seed:.0f}')

            for count,randomize_value in enumerate(self.randomize_value_list):
                
                print(f'random value {randomize_value:.3f}')

                self.train_start_time = time.time()

                # preprocessing #

                self.count = count
                self.randomize_value = randomize_value

                self.df = pd.read_csv(df_name)
                self.unaltered_df = pd.read_csv(df_name)
                self.fill_missing_values()
                    
                self.randomize_data()
                self.split_train_and_test()
                self.impute_data()

                if self.randomize_value == 0:
        
                    my_file = pathlib.Path(f'CSV_scale_target_original\\scale_original_df_validation_feature_seed{self.random_seed}.csv')

                    if my_file.is_file() == False:

                        self.df_validation_feature.to_csv(f'CSV_scale_target_original\\scale_original_df_validation_feature_seed{self.random_seed}.csv',index=False)
                        self.df_train_feature.to_csv(f'CSV_scale_target_original\\scale_original_df_train_feature_seed{self.random_seed}.csv',index=False)
                        self.df_validation_target.to_csv(f'CSV_scale_target_original\\scale_original_df_validation_target_seed{self.random_seed}.csv',index=False)
                        self.df_train_target.to_csv(f'CSV_scale_target_original\\scale_original_df_train_target_seed{self.random_seed}.csv',index=False)

                    self.scale_original_df_validation_feature = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_validation_feature_seed{self.random_seed}.csv') 
                    self.scale_original_df_train_feature = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_train_feature_seed{self.random_seed}.csv') 
                    self.scale_original_df_validation_target = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_validation_target_seed{self.random_seed}.csv') 
                    self.scale_original_df_train_target = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_train_target_seed{self.random_seed}.csv')    

                self.scale_back_target_train_max = self.df_train_target.max().values[0]
                self.scale_back_target_train_min = self.df_train_target.min().values[0]

                self.scale_data()                 
                    
                if self.randomize_value == 0:
    
                    my_file = pathlib.Path(f'CSV\\original_df_validation_feature_seed{self.random_seed}.csv')

                    if my_file.is_file() == False:

                        self.df_validation_feature.to_csv(f'CSV\\original_df_validation_feature_seed{self.random_seed}.csv',index=False)
                        self.df_train_feature.to_csv(f'CSV\\original_df_train_feature_seed{self.random_seed}.csv',index=False)
                        self.df_validation_target.to_csv(f'CSV\\original_df_validation_target_seed{self.random_seed}.csv',index=False)
                        self.df_train_target.to_csv(f'CSV\\original_df_train_target_seed{self.random_seed}.csv',index=False)

                    self.original_df_validation_feature = pd.read_csv(f'CSV\\original_df_validation_feature_seed{self.random_seed}.csv') 
                    self.original_df_train_feature = pd.read_csv(f'CSV\\original_df_train_feature_seed{self.random_seed}.csv') 
                    self.original_df_validation_target = pd.read_csv(f'CSV\\original_df_validation_target_seed{self.random_seed}.csv') 
                    self.original_df_train_target = pd.read_csv(f'CSV\\original_df_train_target_seed{self.random_seed}.csv')    

                # model #

                try:

                    self.current_model = self.model(
                        random_state=np.random.RandomState(0)
                    )

                except:

                    self.current_model = self.model()

                self.train_model()
                self.predict()

                self.train_end_time = time.time()

                self.append_to_dataframe()

            self.combined_df_list.append(self.combined_df)

        self.export_df = pd.concat(self.combined_df_list,axis='columns')

    def fill_missing_values(self):
        
        self.df = self.df.fillna(0)
        self.unaltered_df = self.unaltered_df.fillna(0)

    # SECTION data has categorical data #

    def separate_categorical_and_numerical_data(self):

        self.numerical_df = self.df.select_dtypes(include=[np.number])
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])

    def randomize_categorical_data(self):

        # up to randomize_value percent of data is mislabeled #

        for column in self.categorical_df:
            sample = self.categorical_df[[column]].sample(frac=self.randomize_value,random_state=np.random.RandomState(0))
            sample = sample.reset_index(drop=True)
            self.categorical_df[column].loc[sample.index] = sample.values.ravel()

    def encode_categorical_data(self):

        # SECTION MANUAL CHANGE #

        self.ordinal_categorical_df = self.categorical_df[ordinal_list]
        self.nominal_categorical_df = self.categorical_df[nominal_list]

        self.ordinal_encoder = OrdinalEncoder(categories=categories_list)
        self.ordinal_encoder.fit(self.ordinal_categorical_df)
        self.encoded_ordinal_data = self.ordinal_encoder.transform(self.ordinal_categorical_df)

        self.nominal_encoder = OneHotEncoder()
        self.nominal_encoder.fit(self.nominal_categorical_df)
        self.encoded_nominal_data = self.nominal_encoder.transform(self.nominal_categorical_df).toarray()
        
        nominal_data_unique_value_list = []
        for column in self.nominal_categorical_df:
            for value in self.nominal_categorical_df[column].unique():
                nominal_data_unique_value_list.append(f'{column}_{value}')

        self.encoded_ordinal_df = pd.DataFrame(self.encoded_ordinal_data,columns=self.ordinal_categorical_df.columns)
        self.encoded_nominal_df = pd.DataFrame(self.encoded_nominal_data,columns=nominal_data_unique_value_list)

        self.encoded_categorical_df = pd.concat([self.encoded_ordinal_df,self.encoded_nominal_df],axis='columns')

    def randomize_numerical_data(self):

        if self.randomize == 'FEATURE':

            self.numerical_df_feature = self.numerical_df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            self.numerical_df_target = self.numerical_df[[target_name]]

        if self.randomize == 'TARGET':

            self.numerical_df_feature = self.numerical_df.drop([target_name],axis='columns')
            self.numerical_df_target = self.numerical_df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            
        if self.randomize =='FEATURETARGET':

            self.numerical_df_feature = self.numerical_df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            self.numerical_df_target = self.numerical_df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))            

    def split_numerical_data(self):

        self.numerical_train_feature_df, self.numerical_validation_feature_df, self.numerical_train_target_df, self.numerical_validation_target_df = train_test_split(self.numerical_df_feature,self.numerical_df_target,test_size=0.2,random_state=self.random_state)

    def scale_numerical_data(self):
    
        # scale based on train set for feature #

        for feature_column in self.numerical_train_feature_df:

            self.feature_column_max = self.numerical_train_feature_df[feature_column].max()
            self.feature_column_min = self.numerical_train_feature_df[feature_column].min()
            self.numerical_train_feature_df[feature_column] = self.numerical_train_feature_df[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))
            self.numerical_validation_feature_df[feature_column] = self.numerical_validation_feature_df[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))

        # scale based on train set for target #

        for target_column in self.numerical_train_target_df:
    
            self.target_column_max = self.numerical_train_target_df[target_column].max()
            self.target_column_min = self.numerical_train_target_df[target_column].min()
            self.numerical_train_target_df[target_column] = self.numerical_train_target_df[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))
            self.numerical_validation_target_df[target_column] = self.numerical_validation_target_df[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))

    def combine_categorical_and_numerical(self):

        self.categorical_train_feature = self.encoded_categorical_df.loc[self.numerical_train_feature_df.index]
        self.df_train_feature = pd.concat([self.numerical_train_feature_df,self.categorical_train_feature],axis='columns')

        self.df_train_target = self.numerical_train_target_df

        self.categorical_validation_feature = self.encoded_categorical_df.loc[self.numerical_validation_feature_df.index]
        self.df_validation_feature = pd.concat([self.numerical_validation_feature_df,self.categorical_validation_feature],axis='columns')

        self.df_validation_target = self.numerical_validation_target_df

    # SECTION data has no categorical data #

    def randomize_data(self):

        if self.randomize == 'FEATURE':

            df_feature = self.df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            df_target = self.df[[target_name]]

        if self.randomize == 'TARGET':

            df_feature = self.df.drop([target_name],axis='columns')
            df_target = self.df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            
        if self.randomize =='FEATURETARGET':

            df_feature = self.df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            df_target = self.df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))  

        self.df = pd.concat([df_feature,df_target],axis='columns')

    def split_train_and_test(self):
    
        self.df_feature = self.df.drop([target_name],axis='columns')
        self.df_target = self.df[[target_name]]      

        self.df_train_feature, self.df_validation_feature, self.df_train_target, self.df_validation_target = train_test_split(self.df_feature,self.df_target,test_size=0.2,random_state=self.random_state)

    def impute_data(self):
        
        from sklearn.neighbors import KNeighborsRegressor

        # train KNN imputor based on train split to avoid data leakage #

        df_impute_missing = self.df_train_feature[self.df_train_feature['DG'] == 0].drop(['CD','CONC'],axis='columns')
        df_impute_non_missing = self.df_train_feature[self.df_train_feature['DG'] != 0].drop(['CD','CONC'],axis='columns')

        df_unused_train_set = self.df_train_feature[['CD','CONC']]

        df_KNN_imputor_train_feature = df_impute_non_missing.drop(['DG'],axis='columns')
        df_KNN_imputor_train_target = df_impute_non_missing['DG']

        KNN_imputor = KNeighborsRegressor(
            n_neighbors=3,
            weights='distance'
        )

        KNN_imputor.fit(df_KNN_imputor_train_feature,df_KNN_imputor_train_target)

        # predict for missing data in train set #
         
        df_impute_missing_feature = df_impute_missing.drop(['DG'],axis='columns')

        imputation_index = df_impute_missing_feature.index
        imputation_prediction = pd.DataFrame(KNN_imputor.predict(df_impute_missing_feature),columns=['DG'])
        imputation_prediction = imputation_prediction.set_index(imputation_index)

        imputed_feature = pd.concat([imputation_prediction,df_impute_missing_feature],axis='columns')

        imputed_and_non_missing_df = pd.concat([imputed_feature,df_impute_non_missing],axis='rows')

        old_index = self.df_train_feature.index

        imputed_and_non_missing_df = pd.concat([imputed_feature,df_impute_non_missing],axis='rows').reindex(old_index)
        imputed_and_non_missing_df = pd.concat([imputed_and_non_missing_df,df_unused_train_set],axis='columns')

        self.df_train_feature = imputed_and_non_missing_df
        self.df_train_target = self.df_train_target

        # predict for missing data in validation set #

        df_impute_missing_validation = self.df_validation_feature[self.df_validation_feature['DG'] == 0].drop(['CD','CONC'],axis='columns')
        df_impute_non_missing_validation = self.df_validation_feature[self.df_validation_feature['DG'] != 0].drop(['CD','CONC'],axis='columns')

        df_unused_validation_set = self.df_validation_feature[['CD','CONC']]
        
        df_impute_missing_feature_validation = df_impute_missing_validation.drop(['DG'],axis='columns')

        imputation_index_validation = df_impute_missing_feature_validation.index
        imputation_prediction_validation = pd.DataFrame(KNN_imputor.predict(df_impute_missing_feature_validation),columns=['DG'])
        imputation_prediction_validation = imputation_prediction_validation.set_index(imputation_index_validation)

        imputed_feature_validation = pd.concat([imputation_prediction_validation,df_impute_missing_feature_validation],axis='columns')

        imputed_and_non_missing_df_validation = pd.concat([imputed_feature_validation,df_impute_non_missing_validation],axis='rows')

        old_index = self.df_validation_feature.index

        imputed_and_non_missing_df_validation = pd.concat([imputed_feature_validation,df_impute_non_missing_validation],axis='rows').reindex(old_index)
        imputed_and_non_missing_df_validation = pd.concat([imputed_and_non_missing_df_validation,df_unused_validation_set],axis='columns')

        self.df_validation_feature = imputed_and_non_missing_df_validation
        self.df_validation_target = self.df_validation_target

    def scale_data(self):
    
        # scale based on train set for feature #

        for feature_column in self.df_train_feature:

            self.feature_column_max = self.df_train_feature[feature_column].max()
            self.feature_column_min = self.df_train_feature[feature_column].min()
            self.df_train_feature[feature_column] = self.df_train_feature[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))
            self.df_validation_feature[feature_column] = self.df_validation_feature[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))

        # scale based on train set for target #

        for target_column in self.df_train_target:
    
            self.target_column_max = self.df_train_target[target_column].max()
            self.target_column_min = self.df_train_target[target_column].min()
            self.df_train_target[target_column] = self.df_train_target[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))
            self.df_validation_target[target_column] = self.df_validation_target[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))

    # SECTION model #

    def train_model(self):

        self.current_model.fit(self.df_train_feature,self.df_train_target.values.ravel())

    def predict(self):

        # use unscaled to predict but turn back based on scaled
        # prediction is scaled based on training data
        # real is scaled based on original

        original_train_target_max = self.scale_original_df_train_target.max().values[0]
        original_train_target_min = self.scale_original_df_train_target.min().values[0]

        self.prediction = pd.DataFrame(self.current_model.predict(self.original_df_validation_feature),columns=[target_name]).set_index(self.original_df_validation_target.index)
        self.prediction = self.prediction.applymap(lambda x: (x * (self.scale_back_target_train_max - self.scale_back_target_train_min)) + self.scale_back_target_train_min)      
        # self.prediction = self.prediction.applymap(lambda x: (x * (original_train_target_max - original_train_target_min)) + original_train_target_min)
        self.real = self.original_df_validation_target[[target_name]].loc[self.original_df_validation_target.index]
        self.real = self.real.applymap(lambda x: (x * (original_train_target_max - original_train_target_min)) + original_train_target_min)

        self.real_minus_prediction = abs(self.real - self.prediction)
        self.average = self.real_minus_prediction.mean().values[0]

    # SECTION results #

    def append_to_dataframe(self):

        self.temp_df = pd.DataFrame([[self.randomize_value,self.average,(self.train_end_time-self.train_start_time)/60]],columns=[f'NOISEPERCENTSEED{self.random_seed}',f'AVGSEED{self.random_seed}',f'TIMESEED{self.random_seed}'])
        self.combined_df = pd.concat([self.combined_df,self.temp_df],axis='rows').reset_index(drop=True)

    def return_dataframe(self):

        print(self.export_df)
        return self.export_df

# SECTION Stacked Model #

class Stacking_Process:
    
    def __init__(self,randomize=None,has_categorical=None):

        self.randomize_value_list = [x*0.01 for x in range(0,110,10)]
        self.combined_df_list = []
        self.has_categorical = has_categorical
        self.randomize = randomize
        self.stack_model_tuple = ('XGB','LGBM','RF','GB','ADA','NN','ELAS','LASS','RIDGE','SVM','KNN','DEC')
        self.stack_model_class_tuple = (XGBRegressor,LGBMRegressor,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,MLPRegressor,ElasticNet,Lasso,Ridge,SVR,KNeighborsRegressor,DecisionTreeRegressor)
        self.process()

    def process(self):

        for random_seed in range(5):

            self.random_seed = random_seed
            self.random_state = np.random.RandomState(self.random_seed)
            self.combined_df = pd.DataFrame(columns=[f'NOISEPERCENTSEED{self.random_seed}',f'AVGSEED{self.random_seed}',f'TIMESEED{self.random_seed}'])
            
            print(f'random seed {random_seed:.0f}')

            for count,randomize_value in enumerate(self.randomize_value_list):
      
                print(f'random value {randomize_value:.3f}')

                self.train_start_time = time.time()

                # preprocessing #

                self.count = count
                self.randomize_value = randomize_value

                self.df = pd.read_csv(df_name)
                self.unaltered_df = pd.read_csv(df_name)
                self.fill_missing_values()
                    
                self.randomize_data()
                self.split_train_and_test()
                self.impute_data()

                if self.randomize_value == 0:
            
                    my_file = pathlib.Path(f'CSV_scale_target_original\\scale_original_df_validation_feature_seed{self.random_seed}.csv')

                    if my_file.is_file() == False:

                        self.df_validation_feature.to_csv(f'CSV_scale_target_original\\scale_original_df_validation_feature_seed{self.random_seed}.csv',index=False)
                        self.df_train_feature.to_csv(f'CSV_scale_target_original\\scale_original_df_train_feature_seed{self.random_seed}.csv',index=False)
                        self.df_validation_target.to_csv(f'CSV_scale_target_original\\scale_original_df_validation_target_seed{self.random_seed}.csv',index=False)
                        self.df_train_target.to_csv(f'CSV_scale_target_original\\scale_original_df_train_target_seed{self.random_seed}.csv',index=False)

                    self.scale_original_df_validation_feature = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_validation_feature_seed{self.random_seed}.csv') 
                    self.scale_original_df_train_feature = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_train_feature_seed{self.random_seed}.csv') 
                    self.scale_original_df_validation_target = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_validation_target_seed{self.random_seed}.csv') 
                    self.scale_original_df_train_target = pd.read_csv(f'CSV_scale_target_original\\scale_original_df_train_target_seed{self.random_seed}.csv')    

                self.scale_back_target_train_max = self.df_train_target.max().values[0]
                self.scale_back_target_train_min = self.df_train_target.min().values[0]

                self.scale_data()

                if self.randomize_value == 0:

                    my_file = pathlib.Path(f'CSV\\original_df_validation_feature_seed{self.random_seed}.csv')

                    if my_file.is_file() == False:

                        self.df_validation_feature.to_csv(f'CSV\\original_df_validation_feature_seed{self.random_seed}.csv',index=False)
                        self.df_train_feature.to_csv(f'CSV\\original_df_train_feature_seed{self.random_seed}.csv',index=False)
                        self.df_validation_target.to_csv(f'CSV\\original_df_validation_target_seed{self.random_seed}.csv',index=False)
                        self.df_train_target.to_csv(f'CSV\\original_df_train_target_seed{self.random_seed}.csv',index=False)

                    self.original_df_validation_feature = pd.read_csv(f'CSV\\original_df_validation_feature_seed{self.random_seed}.csv') 
                    self.original_df_train_feature = pd.read_csv(f'CSV\\original_df_train_feature_seed{self.random_seed}.csv') 
                    self.original_df_validation_target = pd.read_csv(f'CSV\\original_df_validation_target_seed{self.random_seed}.csv') 
                    self.original_df_train_target = pd.read_csv(f'CSV\\original_df_train_target_seed{self.random_seed}.csv')     

                # model #
         
                self.k_fold()   
                self.fit()
                self.predict()

                self.train_end_time = time.time()

                self.append_to_dataframe()

            self.combined_df_list.append(self.combined_df)

        self.export_df = pd.concat(self.combined_df_list,axis='columns')

    def fill_missing_values(self):
        
        self.df = self.df.fillna(0)
        self.unaltered_df = self.unaltered_df.fillna(0)

    # SECTION data has categorical data #

    def separate_categorical_and_numerical_data(self):

        self.numerical_df = self.df.select_dtypes(include=[np.number])
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])

    def randomize_categorical_data(self):

        # up to randomize_value percent of data is mislabeled #

        for column in self.categorical_df:
            sample = self.categorical_df[[column]].sample(frac=self.randomize_value,random_state=np.random.RandomState(0))
            sample = sample.reset_index(drop=True)
            self.categorical_df[column].loc[sample.index] = sample.values.ravel()

    def encode_categorical_data(self):

        # SECTION MANUAL CHANGE #

        self.ordinal_categorical_df = self.categorical_df[ordinal_list]
        self.nominal_categorical_df = self.categorical_df[nominal_list]

        self.ordinal_encoder = OrdinalEncoder(categories=categories_list)
        self.ordinal_encoder.fit(self.ordinal_categorical_df)
        self.encoded_ordinal_data = self.ordinal_encoder.transform(self.ordinal_categorical_df)

        self.nominal_encoder = OneHotEncoder()
        self.nominal_encoder.fit(self.nominal_categorical_df)
        self.encoded_nominal_data = self.nominal_encoder.transform(self.nominal_categorical_df).toarray()
        
        nominal_data_unique_value_list = []
        for column in self.nominal_categorical_df:
            for value in self.nominal_categorical_df[column].unique():
                nominal_data_unique_value_list.append(f'{column}_{value}')

        self.encoded_ordinal_df = pd.DataFrame(self.encoded_ordinal_data,columns=self.ordinal_categorical_df.columns)
        self.encoded_nominal_df = pd.DataFrame(self.encoded_nominal_data,columns=nominal_data_unique_value_list)

        self.encoded_categorical_df = pd.concat([self.encoded_ordinal_df,self.encoded_nominal_df],axis='columns')

    def randomize_numerical_data(self):

        if self.randomize == 'FEATURE':

            self.numerical_df_feature = self.numerical_df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            self.numerical_df_target = self.numerical_df[[target_name]]

        if self.randomize == 'TARGET':

            self.numerical_df_feature = self.numerical_df.drop([target_name],axis='columns')
            self.numerical_df_target = self.numerical_df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            
        if self.randomize =='FEATURETARGET':

            self.numerical_df_feature = self.numerical_df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            self.numerical_df_target = self.numerical_df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))            

    def split_numerical_data(self):

        self.numerical_train_feature_df, self.numerical_validation_feature_df, self.numerical_train_target_df, self.numerical_validation_target_df = train_test_split(self.numerical_df_feature,self.numerical_df_target,test_size=0.2,random_state=self.random_state)

    def scale_numerical_data(self):
    
        # scale based on train set for feature #

        for feature_column in self.numerical_train_feature_df:

            self.feature_column_max = self.numerical_train_feature_df[feature_column].max()
            self.feature_column_min = self.numerical_train_feature_df[feature_column].min()
            self.numerical_train_feature_df[feature_column] = self.numerical_train_feature_df[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))
            self.numerical_validation_feature_df[feature_column] = self.numerical_validation_feature_df[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))

        # scale based on train set for target #

        for target_column in self.numerical_train_target_df:
    
            self.target_column_max = self.numerical_train_target_df[target_column].max()
            self.target_column_min = self.numerical_train_target_df[target_column].min()
            self.numerical_train_target_df[target_column] = self.numerical_train_target_df[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))
            self.numerical_validation_target_df[target_column] = self.numerical_validation_target_df[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))

    def combine_categorical_and_numerical(self):

        self.categorical_train_feature = self.encoded_categorical_df.loc[self.numerical_train_feature_df.index]
        self.df_train_feature = pd.concat([self.numerical_train_feature_df,self.categorical_train_feature],axis='columns')

        self.df_train_target = self.numerical_train_target_df

        self.categorical_validation_feature = self.encoded_categorical_df.loc[self.numerical_validation_feature_df.index]
        self.df_validation_feature = pd.concat([self.numerical_validation_feature_df,self.categorical_validation_feature],axis='columns')

        self.df_validation_target = self.numerical_validation_target_df

    # SECTION data has no categorical data #

    def randomize_data(self):

        if self.randomize == 'FEATURE':

            df_feature = self.df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            df_target = self.df[[target_name]]

        if self.randomize == 'TARGET':

            df_feature = self.df.drop([target_name],axis='columns')
            df_target = self.df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            
        if self.randomize =='FEATURETARGET':

            df_feature = self.df.drop([target_name],axis='columns').applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))
            df_target = self.df[[target_name]].applymap(lambda x: x + (x * np.random.uniform(-self.randomize_value,self.randomize_value)))  

        self.df = pd.concat([df_feature,df_target],axis='columns')

    def split_train_and_test(self):
    
        self.df_feature = self.df.drop([target_name],axis='columns')
        self.df_target = self.df[[target_name]]      

        self.df_train_feature, self.df_validation_feature, self.df_train_target, self.df_validation_target = train_test_split(self.df_feature,self.df_target,test_size=0.2,random_state=self.random_state)

    def impute_data(self):
        
        from sklearn.neighbors import KNeighborsRegressor

        # train KNN imputor based on train split to avoid data leakage #

        df_impute_missing = self.df_train_feature[self.df_train_feature['DG'] == 0].drop(['CD','CONC'],axis='columns')
        df_impute_non_missing = self.df_train_feature[self.df_train_feature['DG'] != 0].drop(['CD','CONC'],axis='columns')

        df_unused_train_set = self.df_train_feature[['CD','CONC']]

        df_KNN_imputor_train_feature = df_impute_non_missing.drop(['DG'],axis='columns')
        df_KNN_imputor_train_target = df_impute_non_missing['DG']

        KNN_imputor = KNeighborsRegressor(
            n_neighbors=3,
            weights='distance'
        )

        KNN_imputor.fit(df_KNN_imputor_train_feature,df_KNN_imputor_train_target)

        # predict for missing data in train set #
         
        df_impute_missing_feature = df_impute_missing.drop(['DG'],axis='columns')

        imputation_index = df_impute_missing_feature.index
        imputation_prediction = pd.DataFrame(KNN_imputor.predict(df_impute_missing_feature),columns=['DG'])
        imputation_prediction = imputation_prediction.set_index(imputation_index)

        imputed_feature = pd.concat([imputation_prediction,df_impute_missing_feature],axis='columns')

        imputed_and_non_missing_df = pd.concat([imputed_feature,df_impute_non_missing],axis='rows')

        old_index = self.df_train_feature.index

        imputed_and_non_missing_df = pd.concat([imputed_feature,df_impute_non_missing],axis='rows').reindex(old_index)
        imputed_and_non_missing_df = pd.concat([imputed_and_non_missing_df,df_unused_train_set],axis='columns')

        self.df_train_feature = imputed_and_non_missing_df
        self.df_train_target = self.df_train_target

        # predict for missing data in validation set #

        df_impute_missing_validation = self.df_validation_feature[self.df_validation_feature['DG'] == 0].drop(['CD','CONC'],axis='columns')
        df_impute_non_missing_validation = self.df_validation_feature[self.df_validation_feature['DG'] != 0].drop(['CD','CONC'],axis='columns')

        df_unused_validation_set = self.df_validation_feature[['CD','CONC']]
        
        df_impute_missing_feature_validation = df_impute_missing_validation.drop(['DG'],axis='columns')

        imputation_index_validation = df_impute_missing_feature_validation.index
        imputation_prediction_validation = pd.DataFrame(KNN_imputor.predict(df_impute_missing_feature_validation),columns=['DG'])
        imputation_prediction_validation = imputation_prediction_validation.set_index(imputation_index_validation)

        imputed_feature_validation = pd.concat([imputation_prediction_validation,df_impute_missing_feature_validation],axis='columns')

        imputed_and_non_missing_df_validation = pd.concat([imputed_feature_validation,df_impute_non_missing_validation],axis='rows')

        old_index = self.df_validation_feature.index

        imputed_and_non_missing_df_validation = pd.concat([imputed_feature_validation,df_impute_non_missing_validation],axis='rows').reindex(old_index)
        imputed_and_non_missing_df_validation = pd.concat([imputed_and_non_missing_df_validation,df_unused_validation_set],axis='columns')

        self.df_validation_feature = imputed_and_non_missing_df_validation
        self.df_validation_target = self.df_validation_target

    def scale_data(self):
    
        # scale based on train set for feature #

        for feature_column in self.df_train_feature:

            self.feature_column_max = self.df_train_feature[feature_column].max()
            self.feature_column_min = self.df_train_feature[feature_column].min()
            self.df_train_feature[feature_column] = self.df_train_feature[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))
            self.df_validation_feature[feature_column] = self.df_validation_feature[feature_column].apply(lambda x: (x-self.feature_column_min)/(self.feature_column_max-self.feature_column_min))

        # scale based on train set for target #

        for target_column in self.df_train_target:
    
            self.target_column_max = self.df_train_target[target_column].max()
            self.target_column_min = self.df_train_target[target_column].min()
            self.df_train_target[target_column] = self.df_train_target[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))
            self.df_validation_target[target_column] = self.df_validation_target[target_column].apply(lambda x: (x-self.target_column_min)/(self.target_column_max-self.target_column_min))
        
    # SECTION Model #

    def k_fold(self):
    
        self.number_of_folds = 10
        kfold_holder_for_training = KFold(n_splits=self.number_of_folds,random_state=None,shuffle=False)
        self.splits_for_training = kfold_holder_for_training.split(self.df_train_feature)

    def fit(self):

        self.model_dict = {}
        for model in self.stack_model_tuple:
            self.model_dict[model] = {}

        self.prediction_list = []
        self.META_hold_out_prediction_list = []

        for count_for_train, (train_index_for_validation,test_index_for_validation) in enumerate(self.splits_for_training):

            print(f'Fold {count_for_train}')

            # SECTION splits #

            kfold_train_feature_df = self.df_train_feature.iloc[train_index_for_validation]
            kfold_train_target_df = self.df_train_target.iloc[train_index_for_validation]

            kfold_test_feature_df = self.df_train_feature.iloc[test_index_for_validation]
            kfold_test_target_df = self.df_train_target.iloc[test_index_for_validation]

            for count,model_name in enumerate(self.stack_model_tuple):

                self.model_dict[model_name] = {}

                try:

                    model = self.stack_model_class_tuple[count](
                        random_state=np.random.RandomState(0),
                    )
                
                except:

                    model = self.stack_model_class_tuple[count](
                    )

                model.fit(kfold_train_feature_df,kfold_train_target_df.values.ravel())
                model_prediction_for_training_meta_model = model.predict(kfold_test_feature_df)

                self.model_dict[model_name][count_for_train] = {}
                self.model_dict[model_name][count_for_train]['model'] = model
                self.model_dict[model_name][count_for_train]['meta_feature'] = model_prediction_for_training_meta_model                

            # SECTION stacking #

            META_hold_out_prediction_df = pd.DataFrame()

            for count,key in enumerate(self.model_dict):

                meta_feature_df = pd.DataFrame(self.model_dict[key][count_for_train]['meta_feature'],columns=[f'MODEL {count}'])
                META_hold_out_prediction_df = pd.concat([META_hold_out_prediction_df,meta_feature_df],axis='columns')

            self.META_hold_out_prediction_list.append(META_hold_out_prediction_df)

        self.META_train_df = pd.concat(self.META_hold_out_prediction_list,axis='rows').reset_index(drop=True)

        # SECTION META model #

        META_model = LinearRegression()

        META_model.fit(self.META_train_df,self.df_train_target.values.ravel())

        self.META_model = META_model

        # SECTION refit base models #

        self.refit_base_model_dict = {}

        for count,model_name in enumerate(self.stack_model_tuple):
            
            self.refit_base_model_dict[f'refit{model_name}'] = {}

            try:

                model = self.stack_model_class_tuple[count](
                    random_state=np.random.RandomState(0),
                )
            
            except:

                model = self.stack_model_class_tuple[count](
                )

            model.fit(self.df_train_feature,self.df_train_target.values.ravel())
            model_prediction = pd.DataFrame(model.predict(self.df_validation_feature),columns=[f'MODEL {count}'])

            self.refit_base_model_dict[f'refit{model_name}']['model'] = model

    def predict(self):

        original_train_target_max = self.scale_original_df_train_target.max().values[0]
        original_train_target_min = self.scale_original_df_train_target.min().values[0]  

        self.META_prediction_df = pd.DataFrame()

        for count,refit_model in enumerate(self.refit_base_model_dict):

            refit_model_prediction = pd.DataFrame(self.refit_base_model_dict[refit_model]['model'].predict(self.original_df_validation_feature),columns=[f'MODEL {count}'])
            self.META_prediction_df = pd.concat([self.META_prediction_df,refit_model_prediction],axis='columns')

        self.META_prediction = pd.DataFrame(self.META_model.predict(self.META_prediction_df),columns=['CAP'])

        # self.META_prediction_unscaled = self.META_prediction.applymap(lambda x: (x * (original_train_target_max - original_train_target_min)) + original_train_target_min)
        # self.prediction = self.META_prediction_unscaled.set_index(self.original_df_validation_target.index)

        self.prediction = self.META_prediction.set_index(self.original_df_validation_target.index)
        self.prediction = self.prediction.applymap(lambda x: (x * (self.scale_back_target_train_max - self.scale_back_target_train_min)) + self.scale_back_target_train_min)
        self.real = self.original_df_validation_target[[target_name]].loc[self.original_df_validation_target.index]
        self.real = self.real.applymap(lambda x: (x * (original_train_target_max - original_train_target_min)) + original_train_target_min)

        self.real_minus_prediction = abs(self.real - self.prediction)
        self.average = self.real_minus_prediction.mean().values[0]

    def append_to_dataframe(self):

        self.temp_df = pd.DataFrame([[self.randomize_value,self.average,(self.train_end_time-self.train_start_time)/60]],columns=[f'NOISEPERCENTSEED{self.random_seed}',f'AVGSEED{self.random_seed}',f'TIMESEED{self.random_seed}'])
        self.combined_df = pd.concat([self.combined_df,self.temp_df],axis='rows').reset_index(drop=True)

    def return_dataframe(self):

        print(self.export_df)
        return self.export_df

if run == 'Standalone':

    # SECTION Standalone Model Process #

    model_list = [
        ['LGBM',LGBMRegressor],
        ['XGB',XGBRegressor],
        ['NN',MLPRegressor],
        ['ELAS',ElasticNet],
        ['LASS',Lasso],
        ['RIDGE',Ridge],
        ['RF',RandomForestRegressor],
        ['SVM',SVR],
        ['KNN',KNeighborsRegressor],
        ['GB',GradientBoostingRegressor],
        ['ADA',AdaBoostRegressor],
        ['DEC',DecisionTreeRegressor],
    ]

    randomize_list = ['FEATURE','TARGET','FEATURETARGET']

    for randomize in randomize_list:

        for count in range(len(model_list)):
            
            model_name = model_list[count][0]
            model = model_list[count][1]

            model_process = Process(
                model=model,
                model_name=model_name,
                randomize=randomize,
                has_categorical=has_categorical
            )
            
            model_df = model_process.return_dataframe()
            
            model_df.to_csv(f'CSV_models\\{randomize}{model_name}.csv',index=False)

if run == 'Stack':

    # SECTION Stacked Model Process #

    randomize_list = ['FEATURE','TARGET','FEATURETARGET']

    for randomize in randomize_list:

        model_process = Stacking_Process(
            randomize=randomize,
            has_categorical=has_categorical
        )
        
        model_df = model_process.return_dataframe()
        
        model_df.to_csv(f'CSV_models\\{randomize}STACK.csv',index=False)

if run == 'Both':

    # SECTION Standalone Model Process #

    model_list = [
        ['LGBM',LGBMRegressor],
        ['XGB',XGBRegressor],
        ['NN',MLPRegressor],
        ['ELAS',ElasticNet],
        ['LASS',Lasso],
        ['RIDGE',Ridge],
        ['RF',RandomForestRegressor],
        ['SVM',SVR],
        ['KNN',KNeighborsRegressor],
        ['GB',GradientBoostingRegressor],
        ['ADA',AdaBoostRegressor],
        ['DEC',DecisionTreeRegressor],
    ]

    randomize_list = ['FEATURE','TARGET','FEATURETARGET']

    for randomize in randomize_list:

        for count in range(len(model_list)):
            
            model_name = model_list[count][0]
            model = model_list[count][1]

            model_process = Process(
                model=model,
                model_name=model_name,
                randomize=randomize,
                has_categorical=has_categorical
            )
            
            model_df = model_process.return_dataframe()
            
            model_df.to_csv(f'CSV_models\\{randomize}{model_name}.csv',index=False)

    # SECTION Stacked Model Process #

    randomize_list = ['FEATURE','TARGET','FEATURETARGET']

    for randomize in randomize_list:

        model_process = Stacking_Process(
            randomize=randomize,
            has_categorical=has_categorical
        )
        
        model_df = model_process.return_dataframe()
        
        model_df.to_csv(f'CSV_models\\{randomize}STACK.csv',index=False)


# AT SCALED DATA save data before scaled of randomized model