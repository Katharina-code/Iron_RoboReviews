#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import transformers
import torch
import datasets
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, pipeline, Trainer, TrainingArguments, AutoModel, DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk


# In[2]:


# Load reviews for different categories from McAuley Labs Amazon dataset 
# Due to time and memory constraints, only one category was chosen
reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Appliances")


# In[3]:


# Load meta data for the category
meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Appliances")


# In[23]:


# Print the dataset_dict to view its structure
print(reviews)
print(meta)


# In[26]:


# Convert to pandas dataframe
reviews_df = reviews["full"].to_pandas()
meta_df = meta["full"].to_pandas()


# In[27]:


# Preview reviews dataset
print(reviews_df.head())


# In[28]:


# Preview meta dataset
print(meta_df.head())


# In[31]:


# View column names of both datasets
print(f"Reviews columns: {reviews_df.columns}")
print(f"Meta columns: {meta_df.columns}")


# In[29]:


# Remove all columns that won't be needed in any of the models, to make dataset smaller

# Drop columns from reviews data
reviews_cut = reviews_df.drop (columns = [
    "timestamp", 
    "helpful_vote",
    "images", 
    "asin", 
    "verified_purchase",
    "user_id"
    ])

# Drop columns from meta data
meta_cut = meta_df.drop (columns = [
    "images", 
    "videos", 
    "store",  
    "author"
    ])


# In[32]:


# View column names of both datasets after dropping
print(f"Reviews columns: {reviews_cut.columns}")
print(f"Meta columns: {meta_cut.columns}")


# In[33]:


# Save dataframes to pickle files
reviews_cut.to_pickle("./Saved datasets/reviews.pkl")
meta_cut.to_pickle("./Saved datasets/meta.pkl")


# In[16]:


# Load meta data from the pickle file after restarting machine
reviews_df = pd.read_pickle("./Saved datasets/reviews.pkl")
metad_df = pd.read_pickle("./Saved datasets/meta.pkl")


# In[4]:


# Get the number of columns and rows for reviews dataframe
# Meta data not needed for sentiment analysis
print(f"Shape reviews: {reviews_df.shape}")


# In[6]:


# View final column names of dataset
print(f"Reviews columns: {reviews_df.columns}")


# In[7]:


# Check for missing values
print(reviews_df.isnull().sum())
print(reviews_df.isna().sum())


# In[17]:


# Rename final version of reviews data
reviews_clean = reviews_df


# In[9]:


# Show distirbution between product ratings

# Calculate the frequency of each rating
rating_counts = reviews_clean["rating"].value_counts().sort_index()

# Visualize the distribution using Seaborn
plt.figure(figsize=(5, 3))
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.show()


# In[10]:


# Select 5 random reviews from each rating and inspect the text
samples = reviews_clean.groupby("rating").apply(lambda x: x.sample(min(5, len(x)))).reset_index(drop=True)
print(samples[["rating", "text"]])


# In[18]:


# Use ratings as an approximation for labels (1-2 stars = neg, 3 stars = neut, 4-5 stars = pos)

# Define a function to classify ratings
def classify_rating(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

# Apply the function to create a new "sentiment" column
reviews_clean["sentiment"] = reviews_clean["rating"].apply(classify_rating)

# Display the data with the new column
print(reviews_clean.head())



# In[19]:


# Combine "title" and "text" columns into a "combined_text" column
reviews_clean["combined_text"] = reviews_clean.apply(lambda row: f"{row['title']} {row['text']}", axis=1)


# In[20]:


# Inspect new dataframe
print(reviews_clean.head())


# In[21]:


# Model expects input with sequence length = 512
# Reviews have to be truncated

def truncate_by_characters(text, max_chars=512):
    return text[:max_chars]

reviews_clean['truncated'] = reviews_clean['combined_text'].apply(truncate_by_characters)

print(reviews_clean.head())


# In[22]:


# Load DistilBert model - tokenisation is handled internally
model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student" 
model = AutoModel.from_pretrained(model_name)


# In[23]:


# Check for GPU availability
print("Is CUDA available?", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[25]:


# Model cannot easily be retrained due to being student-teacher model 
# Set up the pipeline using the model card's example
distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=1,
)

# Use the pipeline
result = distilled_student_sentiment_classifier("This is a great product!")
print(result)


# In[26]:


# Function to get the predicted label from the model's output
def get_predicted_label(review):
    prediction_output = distilled_student_sentiment_classifier(review)[0][0]
    predicted_label = prediction_output['label']
    return predicted_label


# In[24]:


# Uncomment if running on GPU
#torch.cuda.empty_cache() 


# In[3]:


# Setting up testing checkpoints to save predictions in batches
chunk_size = 5000  # Define your batch size
checkpoint_file = './Model outputs/sentiment_checkpoint.csv'


# In[ ]:


# Function to save checkpoint
def save_checkpoint(df, filepath):
    df.to_csv(filepath, index=False)

# Check if there's already a checkpoint to load
try:
    checkpoint_df = pd.read_csv(checkpoint_file)
    start_index = checkpoint_df.shape[0]
    processed_reviews = checkpoint_df
except FileNotFoundError:
    start_index = 0
    processed_reviews = pd.DataFrame()

# Process each chunk
for start in range(start_index, len(reviews_clean), chunk_size):
    chunk = reviews_clean.iloc[start:start + chunk_size]
    chunk['predicted_label'] = chunk['truncated'].apply(get_predicted_label)
    
    processed_reviews = pd.concat([processed_reviews, chunk])
    
    # Save the current state as a checkpoint
    save_checkpoint(processed_reviews, checkpoint_file)

print("Processing completed. Final results saved.")


# In[4]:


# Load from csv
processed_reviews = pd.read_csv(checkpoint_file)

# Display the first few rows to verify the content
print(processed_reviews.head())


# In[5]:


# Calculate accuracy
accuracy = accuracy_score(processed_reviews['sentiment'], processed_reviews['predicted_label'])
print(f"Accuracy: {accuracy}")


# In[9]:


# Inspect examples where true != predicted value
mismatches = processed_reviews[processed_reviews['predicted_label'] != processed_reviews['sentiment']]

# Count total mismatches
total_mismatches = mismatches.shape[0]

# Print the total number of mismatches
print(f"Total number of mismatches: {total_mismatches}")


# In[10]:


# Sample 10 random mismatches
random_samples = mismatches.sample(n=10, random_state=42) 

# Inspect the samples
print(random_samples[['truncated', 'sentiment', 'predicted_label']])


# In[8]:


# Print confusion matrix
category_names = ["positive", "negative", "neutral"]

true_labels = processed_reviews['sentiment']
predicted_labels = processed_reviews['predicted_label']

# Step 2: Generate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Step 3: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Step 4: Customize the plot
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

# Set up the tick labels if you have specific category names
ax.set_xticklabels(category_names, rotation=45, ha='right')
ax.set_yticklabels(category_names)

plt.tight_layout()
plt.show()

