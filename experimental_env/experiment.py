import approaches
from dao import DAO
import pandas as pd
import requests
import time
 
"""
Defines the Experiment class, which orchestrates the setup, execution, and logging of machine learning experiments.
Handles data loading, preprocessing, batch settings, and experiment lifecycle.
"""
class Experiment:
    DB_NAME = "experiments.db"

    def __init__(self, dataset_path, columns=['title', 'abstract'], batch_delay=0):
        """
        Initialize an Experiment object.
        Args:
            dataset_path: Path to the dataset CSV file.
            columns: List of feature columns to use.
            batch_delay: Delay (in seconds) between batches.
        """
        self.columns = columns
        # Normalize path separators for cross-platform compatibility
        dataset_path.replace('\\', '/')
        # Extract dataset file name from path
        self._dataset_name = dataset_path[dataset_path.rfind('/') + 1:]
        # Load dataset as DataFrame
        data_df = pd.read_csv(dataset_path)
        # If the dataset provides OpenAlex IDs, fetch article metadata from the API
        if 'openalex_id' in data_df.columns:
            self.df = self.fetch_corpus(data_df)
        # Otherwise, ensure required columns exist and fill missing values
        elif 'title' in data_df.columns and 'abstract' in data_df.columns:
            self.df = data_df.fillna({'title': '', 'abstract': ''})
        else:
            # Raise error if required columns are missing
            raise ValueError("Dataset must contain 'openalex_id' or 'title' and 'abstract' columns")

        # Ensure all record IDs are lowercase strings
        self.df["record_id"] = self.df["record_id"].astype(str).str.lower()

        # Set default batch and API settings
        self.train_batch_size = 'full'
        self.predict_batch_size = 'full'
        self.batch_delay = 0
        self.columns = columns
        self._api_key = None  # initialize the variable

    def set_initial_data(self, positives=1, negatives=1):
        """
        Select initial positive and negative samples for training, and set up inference and label data.
        Args:
            positives: Number of initial positive samples, 1 by default.
            negatives: Number of initial negative samples, 1 by default.
        """
        self.n_initial_positives = positives
        self.n_initial_negatives = negatives
        # Select n positive samples and remove them from the dataset
        remaining_data, positive_data = self._grab_positives(self.df, positives)
        # Select n negative samples and remove them from the dataset
        remaining_data, negative_data = self._grab_negatives(remaining_data, negatives)
        # Store positive and negative samples for training
        self._positive_data = positive_data[['record_id'] + self.columns]
        self._negative_data = negative_data[['record_id'] + self.columns]
        # The rest is used for inference and labeling
        self.inference_X = remaining_data[['record_id'] + self.columns]
        self.labels = remaining_data[['record_id', "label_included"]]

    def set_batch_settings(self, max_train_batch_size='full', max_predict_batch_size='full', batch_delay=0):
        """
        Set batch sizes and delay for training and prediction.
        Args:
            max_train_batch_size: Maximum training batch size.
            max_predict_batch_size: Maximum prediction batch size.
            batch_delay: Delay (in seconds) between batches.
        """
        # Set training and prediction batch sizes and delay
        self.train_batch_size = max_train_batch_size
        self.predict_batch_size = max_predict_batch_size
        self.batch_delay = batch_delay

    def save(self):
        """Save experiment results and metadata to the database via DAO."""
        # Create DAO object and save experiment results to the database
        dao = DAO(self)
        dao.save()

    def class_imbalance_overview(self):
        """
        Print an overview of class imbalance in the dataset.
        Shows the proportion of positive and negative samples.
        """
        # Count positive and negative samples
        count_1 = (self.df['label_included'] == 1).sum()
        count_0 = (self.df['label_included'] == 0).sum()
        total = self.df.shape[0]
        # Print class distribution
        print(f'total valid rows: {total}, {(count_1 / total):.4f} positives and {(count_0 / total):.4f} negatives')

    def fetch_corpus(self, df):
        """
        Fetch article metadata (title, abstract, keywords) from OpenAlex API for each record in the dataset.
        Args:
            df: DataFrame with at least 'openalex_id' and 'label_included' columns.
        Returns:
            DataFrame with fetched metadata.
        """
        # Prepare lists to collect metadata
        ids = []
        titles = []
        abstracts = []
        labels = []
        keywords = []
        # Iterate over each row and fetch metadata from OpenAlex API
        for index, row in df.iterrows():
            try:
                link = row["openalex_id"]
                label = row["label_included"]
                # Extract OpenAlex work ID from URL
                id = link[link.rfind('/') + 1:]
                # Fetch metadata from OpenAlex
                response = requests.get('https://api.openalex.org/works/' + id).json()
                # Only add if required columns are present in the response
                if response is not None and not ('abstract' in self.columns and response["abstract_inverted_index"] is None) and not ('keywords' in self.columns and response["keywords"] is None):
                    title = response["title"]
                    ids.append(id)
                    titles.append(title)
                    labels.append(label)
                    # Add abstract if requested
                    if 'abstract' in self.columns:
                        abstract = ' '.join(response["abstract_inverted_index"].keys())
                        abstracts.append(abstract)
                    # Add keywords if requested
                    if 'keywords' in self.columns:
                        keywords_str = ' '.join(item['display_name'] for item in response["keywords"])
                        keywords.append(keywords_str)
            except Exception as e:
                # Ignore errors and continue
                pass
        # Build DataFrame from collected metadata
        data = {
            'record_id': ids,
            'title': titles,
            'abstract': abstracts,
        }
        # Add keywords and labels if present
        if keywords:
            data['keywords'] = keywords
        if labels:
            data['label_included'] = labels
        result = pd.DataFrame(data)
        return result


    


    def run(self):
        # Set up approach with model, API key, and prompt
        self._approach.model_str = self._model
        self._approach.api_key = self._api_key
        self._approach.prompt = self._prompt if self._prompt else None

        # Provide data to the approach
        self._approach.set_data(self._positive_data, self._negative_data, self.inference_X)
        # Run the approach and time the execution
        start = time.time()
        self.predictions = self._approach.run(self.labels)
        self.duration = time.time() - start

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
        # Validate and set the model string
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
        # Validate and set the approach using the ApproachConverter
        if not value:
            raise ValueError("Name cannot be empty")
        self._approach = approaches.ApproachConverter.convert(value, self.train_batch_size, self.predict_batch_size, self.batch_delay)





    def _grab_positives(self,df,n):
        # Create empty DataFrame for positives
        positive_data = df.copy().head(0)
        for i in range(n):
            # Select the first positive row
            positive_row = df[df['label_included'] == 1].head(1)
            record_id = positive_row['record_id'].iloc[0]
            # Add to positive_data
            positive_data = pd.concat([positive_data, positive_row], ignore_index=True)
            # Remove from df
            df = df[df['record_id'] != record_id]
        return df, positive_data



    def _grab_negatives(self,df,n):
        # Create empty DataFrame for negatives
        negative_data = df.copy().head(0)
        for i in range(n):
            # Select the first negative row
            negative_row = df[df['label_included'] == 0].head(1)
            record_id = negative_row['record_id'].iloc[0]
            # Add to negative_data
            negative_data = pd.concat([negative_data, negative_row], ignore_index=True)
            # Remove from df
            df = df[df['record_id'] != record_id]
        return df, negative_data

