#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from accelerate import init_empty_weights, infer_auto_device_map
from llama_cpp import Llama


# In[2]:


# Load output reviews and meta datasets with sentiment and category columns from previous models 

# Load reviews from csv
processed_reviews = pd.read_csv('./Models output/sentiment_checkpoint.csv')

# Display the first few rows to verify the content
print(processed_reviews.head())


# In[3]:


# Load meta and meta_cat from pickle
meta = pd.read_pickle("./Saved datasets/meta.pkl")
meta_cat = pd.read_pickle("./Saved datasets/meta_cat.pkl")


# In[4]:


# Merge categories to the meta dataframe 
processed_meta = meta.merge(meta_cat[['title', 'category_new']], on='title', how='left')
print(processed_meta.head())


# In[5]:


# Calculate the ranking of products in each category using average rating and rating number columns

# Mean rating and mean rating count
mean_rating = processed_meta['average_rating'].mean()
mean_rating_count = processed_meta['rating_number'].mean()

# Calculate a ranking score based on the Bayesian average formula
processed_meta['ranking_score'] = (
    (processed_meta['average_rating'] * processed_meta['rating_number'] + mean_rating * mean_rating_count) /
    (processed_meta['rating_number'] + mean_rating_count)
)

print(processed_meta[['title', 'ranking_score']])


# In[6]:


# Only include columns needed for model
reviews_clean = processed_reviews[['parent_asin', 'combined_text', 'sentiment']] # Can use the predicted labels here as well
meta_clean = processed_meta[['parent_asin', 'category_new', 'title', 'ranking_score', 'features']] # Can use the predicted categories here as well


# In[32]:


# Path to the GGUF file 
model_path = "./amber.Q4_K_M.gguf"

# Initialize the model
llm = Llama(
    model_path=model_path,  
    n_ctx=6000,  
    n_threads=8,  
    n_gpu_layers=0 
)


# In[8]:


# Create smaller dataset with only top n products

def get_top_products(df, category=None, top_n=3):
     # Filter by category if provided
    if category:
        df = df[df['category_new'] == category]
    
    # Sort the dataframe by ranking score in descending order
    df_sorted = df.sort_values(by='ranking_score', ascending=False)
    
    # Drop duplicates based on parent_asin to ensure unique products
    df_unique = df_sorted.drop_duplicates(subset=['parent_asin'])
    
    # Get top products (up to top_n) after filtering and sorting
    top_products_per_category = df_unique.head(top_n)
    
    return top_products_per_category


# In[16]:


# Testing
category = 'refrigerators, freezers and ice makers'  
top_n = 3  # Number of top products to include

# Test the function
top_products = get_top_products(meta_clean, category, top_n)
print(top_products)


# In[60]:


# Break down prompt in 3 steps due to context length restrictions

# Step 1: Summarize the features for each product

def summarize_features(llm, product_title, features):
    prompt = f"Summarize the features of the product in 3 sentences '{product_title}':\nFeatures: {features}\n"
    output = llm(prompt, max_tokens=300, stop=["</s>"], echo=False)
    return output['choices'][0]['text']

# Step 2: Summarize the positive reviews

def summarize_positive_reviews(llm, product_title, positive_reviews):
    reviews_text = " ".join(positive_reviews)
    prompt = f"Summarize the top 3 positive reviews for '{product_title}':\nReviews: {reviews_text}\n"
    output = llm(prompt, max_tokens=300, stop=["</s>"], echo=False)
    return output['choices'][0]['text']

#Step 3: Summarise the negative reviews:

def summarize_negative_reviews(llm, product_title, negative_reviews):
    reviews_text = " ".join(negative_reviews)
    prompt = f"Summarize the top 3 negative reviews for '{product_title}':\nReviews: {reviews_text}\n"
    output = llm(prompt, max_tokens=300, stop=["</s>"], echo=False)
    return output['choices'][0]['text']


# In[61]:


# Step 4: Concatenate into the final article
def generate_final_article(llm, products, reviews_df):
    final_article = ""

    # Iterate over each product
    for _, product in products.iterrows():
        parent_asin = product['parent_asin']
        title = product['title']
        features = product['features']

        # Retrieve positive and negative reviews for the product
        product_reviews = reviews_df[reviews_df['parent_asin'] == parent_asin]
        positive_reviews = product_reviews[product_reviews['sentiment'] == 'positive']['combined_text'].tolist()
        negative_reviews = product_reviews[product_reviews['sentiment'] == 'negative']['combined_text'].tolist()

        # Summarize product features, positive reviews, and negative reviews
        features_summary = summarize_features(llm, title, features)
        positive_reviews_summary = summarize_positive_reviews(llm, title, positive_reviews[:5])
        negative_reviews_summary = summarize_negative_reviews(llm, title, negative_reviews[:5])

        # Add the summaries to the final article
        final_article += f"\nProduct: {title}\n"
        final_article += f"Features Summary:\n{features_summary}\n"
        final_article += f"Positive Reviews Summary:\n{positive_reviews_summary}\n"
        final_article += f"Negative Reviews Summary:\n{negative_reviews_summary}\n"

    return final_article


# In[62]:


# Show categories as list, so they can be selected by indexing
categories = meta_clean["category_new"].unique().tolist()
print(categories)


# In[64]:


category = categories[0]  # Choose the desired category via index
top_n = 3  # Number of top products to include

# Get the top products for the given category
top_products = get_top_products(meta_clean, category, top_n)

# Generate the final article
final_article = generate_final_article(llm, top_products, reviews_clean)

# Print the final article
print(final_article)

