import boto3
import pickle

bucket_name = "bucket_name"
pickle_file_key = "file_path"  

s3 = boto3.client("s3")

local_pickle_path = "/tmp/stress_level_models.pkl"
s3.download_file(bucket_name, pickle_file_key, local_pickle_path)

with open(local_pickle_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["xgboost"]
scaler = model_data["scaler"]
encoder = model_data["encoder"]
selected_features = model_data["selected_features"]

print("Model loaded successfully from S3!")
