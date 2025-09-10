
"""
Implements the Data Access Object (DAO) for saving experiment results and metadata to the database.
Handles experiment and prompt persistence, and links predictions to experiments.
"""

import sqlite3
import pandas as pd
from prompt import Prompt 



class DAO:
    DB_NAME = "experiments.db"

    def __init__(self, experiment):
        """
        Initialize DAO with an Experiment instance
        Args:
            experiment: Experiment object to persist
        """
        # Store the experiment object for later persistence
        self.experiment = experiment

    def _ensure_tables(self, cursor):
        """Create required tables if they do not exist."""
        # Create prompts, experiments, and targets tables if they don't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS prompts (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Augmentation TEXT,
            Augmentation_Item_Pattern TEXT,
            Prediction TEXT,
            Prediction_Item_Pattern TEXT,
            Positive_Token TEXT,
            Negative_Token TEXT,
            Prediction_Method TEXT
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS experiments (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Approach TEXT,
            Model TEXT,
            Dataset_Name TEXT,
            Features TEXT,
            Max_Train_Batch_Size TEXT,
            Max_Infer_Batch_Size TEXT,
            N_Initial_Positives INTEGER,
            N_Initial_Negatives INTEGER,
            Prompt_ID INTEGER,
            Batch_Delay REAL,
            Duration_Seconds REAL,
            FOREIGN KEY (Prompt_ID) REFERENCES prompts(ID)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS targets (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            RECORD_ID TEXT,
            Experiment_ID INTEGER,
            Label TEXT,
            Prediction TEXT,
            FOREIGN KEY (Experiment_ID) REFERENCES experiments(ID)
        )''')

    def _insert_prompt(self, cursor):
        """Insert prompt and return its ID."""
        pt = self.experiment._prompt
        cursor.execute("""INSERT INTO prompts (Augmentation, Augmentation_Item_Pattern, Prediction, Prediction_Item_Pattern, Positive_Token, Negative_Token, Prediction_Method)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (pt.augmentation, pt.augmentation_item_pattern, pt.prediction, pt.prediction_item_pattern,
                        pt.positive_token, pt.negative_token, pt.prediction_method.value))
        # Return the new prompt's ID
        return cursor.lastrowid

    def _insert_experiment(self, cursor, prompt_id=None):
        """Insert experiment and return its ID."""
        cursor.execute("""INSERT INTO experiments (Approach, Model, Dataset_Name, Features, Max_Train_Batch_Size, Max_Infer_Batch_Size, N_Initial_Positives, N_Initial_Negatives, Prompt_ID, Batch_Delay, Duration_Seconds)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                       (self.experiment._approach.__class__.__name__, self.experiment._model,
                        self.experiment._dataset_name, str(self.experiment.columns), str(self.experiment.train_batch_size), str(self.experiment.predict_batch_size),
                        self.experiment.n_initial_positives, self.experiment.n_initial_negatives, prompt_id, self.experiment.batch_delay, self.experiment.duration))
        # Return the new experiment's ID
        return cursor.lastrowid

    def _insert_targets(self, cursor, exp_id):
        """Insert prediction/label targets for the experiment."""
        # Merge predictions and labels for the experiment
        merged_df = pd.merge(self.experiment.predictions, self.experiment.labels, on='record_id', how='outer')
        # Insert each record into the targets table
        for _, row in merged_df.iterrows():
            cursor.execute("INSERT INTO targets (RECORD_ID, Experiment_ID, Label, Prediction) VALUES (?, ?, ?, ?)",
                           (row['record_id'], exp_id, row['label_included'], row['prediction']))

    def save(self):
        """
        Save experiment metadata, prompt (if any), and predictions to the database.
        Ensures the database file and required tables exist before saving.
        """
        import os
        # Check if sqlite DB file exists, if not, create it
        if not os.path.exists(self.DB_NAME):
            open(self.DB_NAME, 'a').close()

        # Connect to the SQLite database
        conn = sqlite3.connect(self.DB_NAME)
        cursor = conn.cursor()
        # Ensure all required tables exist
        self._ensure_tables(cursor)
        try:
            prompt_id = None
            # Insert prompt if present
            if self.experiment._prompt is not None:
                prompt_id = self._insert_prompt(cursor)
            # Insert experiment metadata
            exp_id = self._insert_experiment(cursor, prompt_id)
            # Insert predictions/labels for the experiment
            self._insert_targets(cursor, exp_id)
        except Exception as e:
            print("DAO save error:", e)
        finally:
            # Commit and close the connection
            conn.commit()
            conn.close()
            print('saved to', self.DB_NAME)

