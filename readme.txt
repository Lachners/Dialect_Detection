Readme:

Corpus extraction (Twitter_extractor.ipynb): 
	- The main Dataset extracted is collected using the Snscrape Library (https://pypi.org/project/snscrape/).
	  As explained on the Thesis, it extracts tweets from the different backgrounds given both dialects, however,
	  due to the updates in Twitters policy, this method renders obsolete at the current date.

	- If the account is from a Spanish account, it is labelled as “ES”, if not, then its labelled as “ARG”

→ Main Dataset (tweets.csv): 30k instances

Evaluation Set extraction (test.ipynb):
	-	By using the tweepy Library (https://www.tweepy.org/), two extra tweet-collections were extracted
	 	for the purpose of evaluation, using firstly tweets from accounts with a high degree of formal language, and after,
		accounts with informal language. These two collections are then preprocessed and reduced in the same manner as
	 	in the Preprocessing Notebook.

→ NO_NE_tweets_test_formal_preprocessed.csv: 2.1k instances
→ NO_NE_tweets_test_informal_preprocessed.csv: 2k instances
→ tweets_test_formal_preprocessed.csv: 2.1k instances
→ tweets_test_informal_preprocessed.csv: 2k instances

_________________________________________________________________________________________________________________________________________


Preprocessing (Preprocessing.ipynb): 
	-	To use the Preprocessing function, firstly it is recommended to read the csv file as a Dataframe and apply
 		the preprocess to that df. 

	-	Includes lower case, removal of punctuation signs, noisy symbols and URL expressions


	-	In the context of training our models, we reduced the “tweets.csv” file, using df.head(10000) 

→ Reduction Dataset for training (subset_tweets_preprocessed.csv): 10k instances

	-	This notebook includes the general Features of the Reduced Dataset for training
	-	This same preprocessing function is employed for preprocessing the Tweets on the research section

Preprocessing NO NE (Preprocessing_NO_NE.ipynb)
	-	its virtually the same code as the normal Preprocessing but adding a filter for NE,
 		using the NER by Manuel Romero (https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner)

→ Reduced Dataset for training NO NE (sub_tweets_preprocessed_NO_NER.csv): 10k instances
	(might be renamed as subset_tweets_preprocessed_NO_NER.csv in other parts)

_________________________________________________________________________________________________________________________________________


FastText (FastText.ipynb):
	-	To train the model, the Dataset columns must be edited to fit the FastText training format,
 		mainly adding _label_ to each row for the given dialect label, as well as creating a “label_description”
 		column which merges the label and the text. After that the data must be splitted into Training and Testing splits (0.8/0.2)
 		and converted into a Dataset through the “train.to_csv” and “test.to_csv” functions.

	-	Here, the Reduced Dataset (subset_tweets_preprocessed.csv) was implemented for the training
 		through “fasttext.train_supervised(input=Dataset)”


	-	For testing inference the method model.predict(sentence) can be used

FastText NO NE (FastText_NONE.ipynb):
	-	the same as earlier but using the NE free Reduction set (sub_tweets_preprocessed_NO_NER.csv)

_________________________________________________________________________________________________________________________________________


BERT Model for Dialect Detection (DistilBERT_wNE.ipynb): 

	-	To train the model, it is necessary to make sure that the dataset is declared and at the same time,
 		that the right BETO model is declared for tokenization, config and model.

	-	The model is trained through the Text Classification DistilBERT configuration of BETO
 		(dccuchile/distilbert-base-spanish-uncased) using the Reduced Dataset (subset_tweets_preprocessed.csv)
 		with a 0.8/0.2 training-testing ratio. 
		The training works over the text and label columns, thus being the labels encoded,
 		to 1 if it was “ES” or 0 if it was “ARG”.
		 It employes the same architecture (dccuchile/distilbert-base-spanish-uncased) as a tokenizer,configuration and base model.
		The trainer function by the transformer’s library is employed for executing the process itself. 
		The output of the training is stored in the “new_stuff” directory. 
		Using the calculate metrics function, F1 Score is calculated

BETO Model for Dialect Detection without NE (DistilBERT_NoNE.ipynb):

	-	Same as the previous notebook, but implementing the NE free Reduced Dataset (sub_tweets_preprocessed_NO_NER.csv)


BETO Model for Multi-Class categorization (multi-class BERT.ipynb):
	
	-	Same as the previous two ones but trained through the text and background columns instead of the dialect labels
 		by using the Reduced Dataset (subset_tweets_preprocessed.csv). 

	-	The outputs are saved in the “/multiclass_stuff” directory
		These outputs are later used for the word embeddings exposed in the research for Demographically related Vocabulary
 		(found in demographic_vocab)

_________________________________________________________________________________________________________________________________________


Evaluation (Inference_testing.ipynb):

	-	By using the evaluation sets extracted and preprocessed previously (tweets_test_formal_preprocessed.csv,
 		tweets_test_informal_preprocessed.csv) we predict the label for each instance of them using for each both models:
 		the ones trained in DistilBERT_NoNE.ipynb and DistilBERT_wNE.ipynb respectively

	-	Through these 4 possible combinations (formal with NE, formal NE free, informal with NE, informal NE free)
 		we calculate the Precision, Recall and F1 Score for each by comparing the actual label with the prediction
 		using the functions from sklearn to do this.



PyPremise (Pypremise.ipynb): 
	
	-	To run this notebook, it is necessary to provide the proper Dataset, already having Predictions and real Labels
	-	In this notebook we solely implemented the NE free predicted sets, due to them being less noisy and more interpretable

	-	PyPremise is a library that can be used for detecting patterns for the prediction outcomes provided.
		To get these patterns we use the predicted Datasets from evaluation (predicted_NO_NE_tweets_test_formal_preprocessed.csv,
		 predicted_NO_NE_tweets_test_informal_preprocessed.csv) and count the matches (1) and mismatches (0)
 		between predictions and real labels. We then save the tokenized words of each sentence in a list called features,
 		while the PyPremise matching labels are saved in the Py_labels = [] list. 

	-	The outcome shows in which kind of features the model might show issues or perform specifically well.


_________________________________________________________________________________________________________________________________________


Dialect Mixture and Code-Switching (Dialect_Mix.ipynb):

	-	We extract a new collection of Tweets (tweets_dialect_mix.csv) using a very similar code to the one implemented for extracting the Evaluation collection,
		but instead of setting a query through accounts in this case it is through a query formulated through expressions from an Argentinian and Spanish list respectively.
	 	Also, it considers on the query Argentina and Spain as the locations to extract from, using the expressions from the opposite country. To get as much tweets as possible,
 		“Buenos Aires” and “Madrid” were added as well, which will later be integrated into the Argentina or Spain location category later.

	-	The count of each expression on the total is then counted

Preprocessing and Analysis (analysis_dialect_mix.ipynb):

	-	The previously extracted corpus is preprocessed using the same Function as in the Preprocessing Notebook,
 		and the label is predicted for each instance using the DistilBERT model.

	-	Using the outcome of this predictions we plot a graph that uses word counts for each expression and compares it with the proportion of predictions.

→ tweets_dialect_mix.csv: 375 instances

	-	This Notebook also contains the print_sentences_with_word function, which prints all the sentences extracted that contain a input certain word.
 		This function is used to visualize the context in which the expressions might appear.



_________________________________________________________________________________________________________________________________________


Relation between Demographic background and Vocabulary (vocab_demographic.ipynb):

	-	The Notebook for this part of the work uses the preprocessed version of the Main Dataset extracted at the beginning (tweets_preprocessed.csv) for visualizing
 		the Word Occurrence and Mean Occurrence of expressions from an additional list of words within each category of background.
 		A new list is then created and words that do not occur at all are not included in this new list.


	-	Using this new list of words, by using the word embeddings from the BERT models trained previously (BERT Model for Dialect Detection and multi-class BERT),
		and using additionally the Kmeans function by Sklearn to cluster the expressions together. Finally, the results of all this are plotted.

	-	Using an additional list of grammatical features, the embeddings of these expressions are visualized through the same method as before

