import approaches
from prompt import Prompt
import pandas as pd
import json
import requests
import time


class Search:
    """
    Search tool for querying a document corpus using LLMs or classical models.
    Allows optional augmentation with example positive documents (RAG/few-shot style).
    Handles data loading, preprocessing, and batch settings.
    """
    def __init__ (self, dataset_path, columns=['title', 'abstract'], user_input=''):
        self.columns = columns
        dataset_path = dataset_path.replace('\\', '/')
        data_df = pd.read_csv(dataset_path)
        if 'openalex_id' in data_df.columns:
            self.df = self.fetch_corpus(data_df)
        elif all(item in data_df.columns for item in self.columns):
            final_column_set = ['record_id'] + self.columns
            self.df = data_df[final_column_set]
            self.df = self.df.fillna({col: '' for col in self.columns})
        else:
            raise ValueError(f"Dataset must contain 'openalex_id' or {columns} columns")

        self.df["record_id"] = self.df["record_id"].astype(str).str.lower()

        self.user_input = user_input

        # Create a DataFrame with one row
        self._positive_data = pd.DataFrame(
            [{'record_id': '00000', self.columns[0]: self.user_input}],
            columns=['record_id'] + self.columns
        )
        # set default settings
        self.train_batch_size = 'full'
        self.predict_batch_size = 'full'
        self.batch_delay = 0
        self.columns = columns
        self._prompt = None
        self._api_key = None  # initialize the variable

    def set_initial_data(self, augmentation_dataset_path, columns=['title', 'abstract']):
        """
        Add example positive documents from an augmentation dataset (no negatives expected).
        """
        augmentation_dataset_path = augmentation_dataset_path.replace('\\', '/')
        data_df = pd.read_csv(augmentation_dataset_path)
        if 'openalex_id' in columns:
            aug_df = self.fetch_corpus(data_df)
        elif all(item in data_df.columns for item in columns):
            aug_df = data_df.fillna({col: '' for col in data_df.select_dtypes(include='object').columns})
        else:
            raise ValueError(f"Dataset must contain 'openalex_id' or {columns} columns")

        aug_df["record_id"] = aug_df["record_id"].astype(str).str.lower()
        # Append all positive examples to _positive_data
        self._positive_data = pd.concat([self._temp_positive_data, aug_df[['record_id'] + self.columns]], ignore_index=True)

    def set_batch_settings(self,max_train_batch_size='full',max_predict_batch_size='full',batch_delay=0):
        
         self.train_batch_size=max_train_batch_size
         self.predict_batch_size=max_predict_batch_size
         self.batch_delay=batch_delay
         
    


    def fetch_corpus(self, df):
        """
        Fetch article metadata (title, abstract, keywords) from OpenAlex API for each record in the dataset.
        Args:
            df: DataFrame with at least 'openalex_id' column.
        Returns:
            DataFrame with fetched metadata (record_id, title, abstract, and optionally keywords).
        """
        # Prepare lists to collect metadata
        ids = []
        titles = []
        abstracts = []
        keywords = []
        # Iterate over each row and fetch metadata from OpenAlex API
        for index, row in df.iterrows():
            try:
                link = row["openalex_id"]
                # Extract OpenAlex work ID from URL
                id = link[link.rfind('/')+1:]
                # Fetch metadata from OpenAlex
                response = requests.get('https://api.openalex.org/works/' + id).json()
                # Only add if required columns are present in the response
                if response is not None and not ('abstract' in self.columns and response["abstract_inverted_index"] is None) and not ('keywords' in self.columns and response["keywords"] is None):
                    title = response["title"]
                    ids.append(id)
                    titles.append(title)
                    # Add abstract if requested
                    if 'abstract' in self.columns:
                        abstract = ' '.join(response["abstract_inverted_index"].keys())
                        abstracts.append(abstract)
                    # Add keywords if requested
                    if 'keywords' in self.columns:
                        keywords_str = ' '.join(item['display_name'] for item in response["keywords"])
                        keywords.append(keywords_str)
            except Exception as e:
                # Print error and continue
                print(e)
        # Build DataFrame from collected metadata
        data = {
            'record_id': ids,
            'title': titles,
            'abstract': abstracts,
        }
        # Add keywords if present
        if keywords:
            data['keywords'] = keywords
        result = pd.DataFrame(data)
        return result
        

    def run(self):
        """
        Run the search approach over the corpus using the current settings and data.
        Sets up the approach, provides data, and returns predictions or ranked results.
        Returns:
            DataFrame of predictions or ranked results.
        """
        # Set up approach with model, API key, and prompt
        self._approach.model_str = self._model
        self._approach.api_key = self._api_key
        self._approach.prompt = self._prompt if self._prompt else None

        # Provide data to the approach (positives, and corpus)
        self._approach.set_data(self._positive_data, self.df)
        # Run the approach and time the execution
        start = time.time()
        self.predictions = self._approach.run()
        self.duration = time.time() - start

        # Return predictions or ranked results
        return self.predictions
        


    @property
    def prompt(self):
        return self._prompt
    @prompt.setter
    def prompt(self, value):
        self._prompt = value


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._model = value

    @property
    def api_key(self):
        return self._api_key
    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def approach(self):
        return self._approach

    @approach.setter
    def approach(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._approach = approaches.ApproachConverter.convert(value,self.train_batch_size,self.predict_batch_size,self.batch_delay)

