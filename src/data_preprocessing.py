# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fit_combined_encoder(col_name, df1, df2):
    le = LabelEncoder()
    combined_values = pd.concat([df1[col_name], df2[col_name]]).unique()
    le.fit(combined_values)
    return le

def encode_datasets(original_df, pairwise_df):
    label_encoders = {}
    label_encoders['Current Crop'] = fit_combined_encoder('Current Crop', pairwise_df, original_df.rename(columns={'Crop Name': 'Current Crop'}))
    label_encoders['Next Crop'] = fit_combined_encoder('Next Crop', pairwise_df, original_df.rename(columns={'Crop Name': 'Next Crop'}))
    label_encoders['Soil Type'] = fit_combined_encoder('Soil Type', pairwise_df, original_df)
    label_encoders['Current Season'] = fit_combined_encoder('Current Season', pairwise_df, original_df.rename(columns={'Season': 'Current Season'}))
    label_encoders['Next Season'] = fit_combined_encoder('Next Season', pairwise_df, original_df.rename(columns={'Season': 'Next Season'}))

    pairwise_encoded = pairwise_df.copy()
    for col in ['Current Crop', 'Next Crop', 'Soil Type', 'Current Season', 'Next Season']:
        pairwise_encoded[col] = label_encoders[col].transform(pairwise_encoded[col])
    return pairwise_encoded, label_encoders