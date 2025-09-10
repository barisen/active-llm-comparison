from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import models
import re
from abc import  abstractmethod
import math
import time 
import models




class BaseApproach:

    def __init__(self, max_train_batch_size, max_predict_batch_size, batch_delay):
        # Initialize approach parameters and placeholders
        self._prompt = None
        self._model = None
        self._api_key = None
        self.train_batch_size = max_train_batch_size
        self.predict_batch_size = max_predict_batch_size
        self.batch_delay = batch_delay

    def run(self, real_targets):
        """
        Run the approach over the provided targets, handling batching and vectorization.
        Args:
            real_targets: DataFrame of true labels for inference set.
        Returns:
            DataFrame of predictions.
        """
        # Prepare positive and negative labeled data
        df_pos_labeled = self.positive_X.copy()
        df_pos_labeled['label'] = 1  # or "positive"
        df_neg_labeled = self.negative_X.copy()
        df_neg_labeled['label'] = 0  # or "negative"
        # Combine them for training labels and features
        train_y = pd.concat([df_pos_labeled, df_neg_labeled], ignore_index=True)[['record_id', 'label']]
        train_X = pd.concat([self.positive_X, self.negative_X], ignore_index=True)
        inference_X = self.inference_X.copy()
        # Convert model string to model instance
        model = models.ModelConverter.convert(self.model_str, self.api_key, self._prompt)
        # Vectorize if not an LLM
        if not isinstance(model, models.BaseLLM):
            train_X, inference_X = self._vectorize_whole_corpus(train_X.map(self.clean_doc), inference_X.map(self.clean_doc))
        # Determine batch sizes
        train_batch_widnow = train_X.shape[0] if self.train_batch_size == 'full' else self.train_batch_size
        inference_batch_size = inference_X.shape[0] if self.predict_batch_size == 'full' else self.predict_batch_size
        current_train_X = train_X.tail(train_batch_widnow)
        current_train_y = train_y.tail(train_batch_widnow)
        num_full_batches = inference_X.shape[0] // inference_batch_size
        # Process each batch
        for i in range(num_full_batches):
            current_inference = inference_X.iloc[i * inference_batch_size: (i + 1) * inference_batch_size]
            current_real_targets = real_targets.iloc[i * inference_batch_size: (i + 1) * inference_batch_size]
            # Update the batch sizes, because this will change in active learning if it's on 'full'
            train_batch_widnow = current_train_X.shape[0] if self.train_batch_size == 'full' else self.train_batch_size
            inference_batch_size = inference_X.shape[0] if self.predict_batch_size == 'full' else self.predict_batch_size
            # Execute the batch and update predictions
            current_train_X, current_train_y, batch_predictions = self._execute_batch(
                current_train_X.tail(train_batch_widnow),
                current_train_y.tail(train_batch_widnow),
                current_inference, current_real_targets)
            self.predictions = pd.concat([self.predictions, batch_predictions], ignore_index=True)
            time.sleep(self.batch_delay)
        # Handle the leftover (if any)
        remainder = inference_X.shape[0] % inference_batch_size
        if remainder > 0:
            current_inference = inference_X.iloc[-remainder:]
            current_real_targets = real_targets.iloc[-remainder:]
            # Update the current train x and y..this is important for active learning
            current_train_X, current_train_y, batch_predictions = self._execute_batch(
                current_train_X, current_train_y, current_inference, current_real_targets)
            self.predictions = pd.concat([self.predictions, batch_predictions], ignore_index=True)
        # Return all predictions
        return self.predictions
    @property
    def api_key(self):
        """Get the API key for LLMs (if used)."""
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        """Set the API key for LLMs (if used)."""
        self._api_key = value

    @property
    def model_str(self):
        return self._model_str

    @model_str.setter
    def model_str(self, value):
        if value:
            self._model_str = value

    @property
    def prompt(self):
        return self._prompt
    @prompt.setter
    def prompt(self, value):
        self._prompt = value



    def set_data(self,positive_X,negative_X,inference_X):
        # Store the datasets for positive, negative, and inference sets
        self.positive_X=positive_X
        self.negative_X=negative_X
        self.inference_X=inference_X

        # Initialize empty predictions DataFrame
        predictions=self.inference_X.copy()
        predictions['prediction']=np.full(inference_X.shape[0], np.nan)
        self.predictions=predictions[['record_id','prediction']].head(0)

    
    def _vectorize_whole_corpus(self,df1,df2):
        # Step 1: Combine title + abstract for each row, for joint corpus
        df1_texts = df1.drop(columns=['record_id']).fillna('').apply(lambda row: ' '.join(row), axis=1).tolist()
        df2_texts = df2.drop(columns=['record_id']).fillna('').apply(lambda row: ' '.join(row), axis=1).tolist()
        all_texts = df1_texts + df2_texts  # shared text corpus

        # Step 2: Fit joint vectorizer on combined texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Step 3: Split matrix back into df1 and df2
        tfidf_features = vectorizer.get_feature_names_out()
        tfidf_df1 = pd.DataFrame(tfidf_matrix[:len(df1)].toarray(), 
                                columns=[f"tfidf_{feat}" for feat in tfidf_features],
                                index=df1.index)
        tfidf_df2 = pd.DataFrame(tfidf_matrix[len(df1):].toarray(), 
                                columns=[f"tfidf_{feat}" for feat in tfidf_features],
                                index=df2.index)

        # Step 4: Build final DataFrames: keep record_id, replace title & abstract
        df1_final = pd.concat([df1[['record_id']].reset_index(drop=True), tfidf_df1.reset_index(drop=True)], axis=1)
        df2_final = pd.concat([df2[['record_id']].reset_index(drop=True), tfidf_df2.reset_index(drop=True)], axis=1)

        # Return vectorized DataFrames
        return df1_final,df2_final
    

    def stem(self,word):
     if (word[-3:]=="ing") and len(word) > 4 :
            return word[:-3]
     if (word[-2:]=="ed") and len(word) > 3 :
            return word[:-2]
     if (word[-2:]=="es") and len(word) > 3 :
            return word[:-2]
     if (word[-1:]=="s") and len(word) > 2 :
            return word[:-1]
        
     return word
    
    def clean_doc(self,doc):
     STOP_WORDS=[  "a", "an", "the", "is", "in", "at", "of", "on", "and", "or", "for", "with", "to", "from"]
     # Tokenization: Convert to lowercase and split based on non-word characters (remove punctuation, etc.)
     try:
            
       tokens = re.findall(r'\b\w+\b', doc.lower())
       # Stopword removal and stemming
       cleaned_tokens = [self.stem(word) for word in tokens if word not in STOP_WORDS]
       
       return " ".join(cleaned_tokens)
     except Exception as e:
            return ""









class Active(BaseApproach):
    
    stop_after_ratio_negatives=.05
    def _execute_batch(self,train_X,train_y,inference_X,real_targets):
        # Stop after a certain ratio of negatives have been found
        stop_after_n_negatives=int(self.inference_X.shape[0]*self.stop_after_ratio_negatives)
        i_negatives=0
        current_train_X=train_X.copy()
        current_inference_X=inference_X.copy()
        current_train_y= train_y.copy()
        predictions=self.predictions.copy().head(0)

        # Iterate over inference set
        for i in range(inference_X.shape[0]):
            if i_negatives==stop_after_n_negatives:
                break
            # Create and train model for each batch
            model=models.ModelConverter.convert(self.model_str,self.api_key,self._prompt)
            model.train(current_train_X,current_train_y)
            record_id_of_first=model.predict_next(current_inference_X)
            if(record_id_of_first==None):
                #if there is no positive, simply take the first row and predict it negative
                record_id_of_first=str(current_inference_X['record_id'].iloc[0])
                predictions = pd.concat([predictions, pd.DataFrame([{'record_id':record_id_of_first,'prediction':0}])], ignore_index=True)
            else:
                predictions = pd.concat([predictions, pd.DataFrame([{'record_id':record_id_of_first,'prediction':1}])], ignore_index=True)
                
            # Identify the row based on a condition (e.g., record_id)
            labeled_row = current_inference_X[current_inference_X['record_id'] == record_id_of_first]
            true_label=real_targets[real_targets['record_id'] == record_id_of_first]['label_included'].iloc[0]
            # Append labeled row to training set
            current_train_X = pd.concat([current_train_X, labeled_row], ignore_index=True)
            current_train_y = pd.concat([current_train_y, pd.DataFrame([{'record_id':record_id_of_first,'label':true_label}])], ignore_index=True)
            if(true_label==0):
                i_negatives+=1
            # Remove it from current_inference_X
            current_inference_X = current_inference_X[current_inference_X['record_id'] != record_id_of_first]

        # Add prediction = 0 to the rest
        current_inference_X['prediction'] = 0
        rest=current_inference_X[['record_id','prediction']]
        # Combine predictions and the rest
        predictions = pd.concat([predictions, rest], ignore_index=True)
        return current_train_X,current_train_y,predictions





class  LowShot(BaseApproach):
    #Note: real_targets is not used, but it is a parameter, to ensure consistency
    def _execute_batch(self,train_X,train_y,inference_X,real_targets):
        # Train and predict in a single batch (low-shot learning)
        model=models.ModelConverter.convert(self.model_str,self.api_key,self._prompt)
        model.train(train_X,train_y)
        predictions=model.predict(inference_X)
        return train_X,train_y,predictions





class ApproachConverter:
    @staticmethod
    def convert(approach_str,train_batch_size,predict_batch_size,batch_delay):
        # Convert string to appropriate approach class
        approach_str=approach_str.lower()
        if('active' in approach_str):
            return Active(train_batch_size,predict_batch_size,batch_delay)
        elif 'shot' in approach_str:
            return LowShot(train_batch_size,predict_batch_size,batch_delay)
        else:
            return None

