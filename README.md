# Ironhack_RoboReviews

This project uses Amazon 2023 Reviews data for the category "Appliances" and three LLM models from Hugging Face to:
- Build a sentiment analysis model to cateogrise customer reviews into positive, negative and neutral
- Build a categorisation model, that can assign one of six pre-defined categories based on product titles
- Build a text genration model that uses the output of the previous models to write a blog post of the top products for a certain category

Each of the sub-projects has their own jupyter notebook file.
The python files were created for linting with pylint
All code can be found in the "Models code" folder.
1. sentiment -> this is the sentiment alaysis model (this is also where the data is loaded for the first time and much of the pre-processing is done)
2. category -> this is the categorisation model (uses pre-processed data saved to pickle, the model itself is saved under categoriser_model)
3. generate -> this is the text generation model (the model itself was downloaded and saved (amber.Q4_K_M.gguf))

The datasets and model outputs were sometimes saved at certain processing stages, to prevent having to load everything again.
These were not uploaded to the GitHub repo
