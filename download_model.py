from sentence_transformers import SentenceTransformer
import os

# Define the model name and the path where you want to save it
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH = "./models/all-MiniLM-L6-v2"

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Download and save the model
print(f"Downloading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"Saving model to: {SAVE_PATH}...")
model.save(SAVE_PATH)

print("Model downloaded and saved successfully!")
