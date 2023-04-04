# LTR-on-torob-data
## FetchAndPreprocess : 
This part of the project is based on the solution provided by torob
## FeatureExtraction : 
this part is used to preprocess and vectorize text data for a search engine application. The code uses Python's pandas library to read and process data from JSON lines files. Specifically, it reads preprocessed product titles, aggregated search queries, and preprocessed test queries, and tokenizes them using Keras' Tokenizer class. It then pads the sequences to a maximum length and saves them as pickle files.

The first block of code processes the preprocessed product titles. It initializes a Tokenizer object and fits it on the preprocessed product titles. It then converts the text data to sequences and pads them to a maximum length before saving them as a pickle file.

The second block of code processes the aggregated search queries. It reads the data from the aggregated search queries file, tokenizes it using the same Tokenizer object used for products, and pads it to a maximum length. It then saves the processed queries as a pickle file.

The third block of code processes the preprocessed test queries. It reads the data from the preprocessed test queries file, tokenizes it using the same Tokenizer object used for products and queries, and pads it to a maximum length. It then saves the processed queries as a pickle file.

The final block of code uses scikit-learn's TfidfVectorizer to generate TF-IDF features for the preprocessed product titles. It initializes a vectorizer with a specified vocabulary size and fits it on the preprocessed product titles. It then transforms the products and queries to TF-IDF feature vectors and saves them as pickle files.

Overall, this code sets up the preprocessed data and feature vectors that will be used for building the search engine model.

## RankNet : 
The code implementing a learning-to-rank model using a pairwise ranking algorithm called RankNet. The model takes in two vectors, a query vector, and a document vector, concatenates them, and passes them through a neural network consisting of two dense layers. The output of the model is a single sigmoid value representing the probability that the document is relevant to the given query.
