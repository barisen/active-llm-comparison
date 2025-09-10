
import re
import numpy as np
import builtins
import pandas as pd
import json
from enum import Enum
from pydantic import BaseModel




class SafeDict(builtins.dict):
    """
    Dictionary subclass that returns a default value ('') for missing keys.
    Useful for safe string formatting with missing fields.
    """
    def __missing__(self, key):
        # Return empty string for missing keys to avoid KeyError in format_map
        return ''



class Prompt:
    """
    Handles prompt creation, schema validation, and robust parsing for LLM-based experiments.
    Supports different response schemas and methods for handling LLM output.
    """
    class Paper_Token_Only(BaseModel):
        """Schema for responses with only STATUS field."""
        STATUS: str
    class Paper_ID_Only(BaseModel):
        """Schema for responses with only ID field."""
        ID: str
    class Paper_ID_Token(BaseModel):
        """Schema for responses with both ID and STATUS fields."""
        ID: str
        STATUS: str

    def __init__(self, augmentation, augmentation_item_pattern, prediction, prediction_item_pattern,
                 positive_token, negative_token, prediction_method='token'):
        """
        Initialize a Prompt object.
        Args:
            augmentation: Template for augmentation prompt.
            augmentation_item_pattern: Pattern for formatting augmentation items.
            prediction: Template for prediction prompt.
            prediction_item_pattern: Pattern for formatting prediction items.
            positive_token: Token representing positive class.
            negative_token: Token representing negative class.
            prediction_method: 'token', 'id', or 'id_token' for response schema.
        """
        self.augmentation = augmentation
        self.augmentation_item_pattern = augmentation_item_pattern
        self.prediction = prediction
        self.prediction_item_pattern = prediction_item_pattern
        self.positive_token = positive_token
        self.negative_token = negative_token
        self.prediction_method = prediction_method

        # Select response schema based on prediction method
        if self.prediction_method == 'id':
            self.response_schema = self.Paper_ID_Only
        elif self.prediction_method == 'token':
            self.response_schema = self.Paper_Token_Only
        else:
            self.response_schema = self.Paper_ID_Token

    def create_augmentation_prompt(self, augmentation_X, augmentation_y):
        """
        Create a prompt for data augmentation using provided data and template.
        Args:
            augmentation_X: DataFrame of features.
            augmentation_y: DataFrame of labels.
        Returns:
            Formatted augmentation prompt string.
        """
        # Merge features and labels for augmentation
        augmentation_data = pd.merge(augmentation_X, augmentation_y, on='record_id', how='outer')
        # Render each item using the pattern and label token
        rendered_items = ", ".join(
            self.augmentation_item_pattern.format_map(
                SafeDict({**row, 'label_token': self.positive_token if row.get('label') == 1 else self.negative_token}))
            for _, row in augmentation_data.iterrows()
        )
        # Format the final augmentation prompt
        final_prompt = self.augmentation.format(rendered_items)
        return final_prompt

    def create_prediction_prompt(self, novel_data, add_format=False):
        """
        Create a prompt for prediction using provided data and template.
        Args:
            novel_data: DataFrame of new data to predict.
            add_format: Whether to append a JSON format example.
        Returns:
            Formatted prediction prompt string.
        """
        # Render each item for prediction
        rendered_items = ", ".join(self.prediction_item_pattern.format_map(SafeDict(row)) for _, row in novel_data.iterrows())
        # Format the final prediction prompt
        final_prompt = self.prediction.format(rendered_items)
        if add_format:
            # Optionally append a JSON format example for the LLM
            format_str = json.dumps([{f: f for f, d in self.response_schema.schema()['properties'].items()}] * 5, indent=4)
            final_prompt += f"Answer using the following format json list {format_str}"
        return final_prompt

    def read_predictions(self, prediction_answer, inference_data):
        """
        Parse LLM prediction output and map to DataFrame.
        Args:
            prediction_answer: Raw LLM output (string).
            inference_data: DataFrame of inference data.
        Returns:
            DataFrame with predictions.
        """
        # Start with all predictions set to 0
        predictions = inference_data.copy()
        predictions['prediction'] = 0
        predictions = predictions[['record_id', 'prediction']]
        # Try to parse the LLM's output
        data = self.robust_json_parse(prediction_answer)
        if not isinstance(data, list):
            data = [data]
        # Convert all dict values to strings for consistency
        data = [{k: str(v) for k, v in d.items()} for d in data if isinstance(d, dict)]
        if self.prediction_method == Methods.TOKEN:
            # For TOKEN method, extract STATUS and compare to positive_token
            paper_items = self.safe_load_paper_items(data, self.Paper_Token_Only)
            # Extract STATUS values (handle None if needed)
            # Sometimes the LLM responds with the token without special characters, cover this case
            status_list = [1 if (paper.STATUS == self.positive_token or paper.STATUS == re.sub(r'[^A-Za-z0-9 ]', '', self.positive_token)) else 0 for paper in paper_items]
            # Assign as new column in DataFrame
            predictions['prediction'] = status_list[:len(predictions['prediction'])] #make sure the sizes match
        elif (self.prediction_method==Methods.ID):
            # Normalize to list of objects
            paper_items = self.safe_load_paper_items(data, self.Paper_ID_Only)
            for paper in paper_items:
                predictions.loc[predictions['record_id'] == paper.ID, 'prediction'] = 1
        else:
            # Normalize to list of objects
            paper_items = self.safe_load_paper_items(data, self.Paper_ID_Token)
            for paper in paper_items:
                predictions.loc[predictions['record_id'] == paper.ID, 'prediction'] = 1 if paper.STATUS==self.positive_token else 0
        return predictions

        




    def parse_mixed_malformed_ids_and_statuses(self, json_str):
        """
        Parse mixed malformed JSON string containing:
        - repeated "ID": value pairs (even malformed),
        - repeated "STATUS": value pairs (even malformed),
        - standalone values after first key-value pair.
        Returns list of dicts like [{'ID': value}, {'STATUS': value}, ...]
        """
        results = []
        # 1. Extract all "ID": or "STATUS": value pairs (quoted string or number, possibly malformed trailing quote)
        pair_pattern = re.compile(r'"(ID|STATUS)"\s*:\s*(?:"([^"]*)"|(\d+))"?')
        for match in pair_pattern.finditer(json_str):
            key, str_val, num_val = match.groups()
            if str_val is not None:
                results.append({key: str_val})
            elif num_val is not None:
                results.append({key: int(num_val)})
        # 2. Extract the first key-value pair to identify where standalone values start
        first_match = pair_pattern.search(json_str)
        if not first_match:
            # No ID or STATUS keys at all — fallback to empty list
            return results
        # Extract substring after first key-value pair
        after_first = json_str[first_match.end():]
        # 3. Extract standalone values (numbers or quoted strings) separated by commas
        standalone_pattern = re.compile(r'("[^"]+"|\d+)')
        for m in standalone_pattern.finditer(after_first):
            v = m.group(1)
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            if v.isdigit():
                v_int = int(v)
                # Only add if not already present
                if {'ID': v_int} not in results and {'STATUS': v_int} not in results:
                    results.append({'ID': v_int})  # Default to ID if no key
            else:
                if {'ID': v} not in results and {'STATUS': v} not in results:
                    results.append({'ID': v})  # Default to ID if no key
        return results

    def robust_json_parse(self, prediction_answer):
        """
        Try to parse JSON normally, else try regex to extract JSON-like block, else try special malformed parser for mixed cases.
        Returns parsed object or empty list on failure.
        """
        try:
            # Try normal JSON parsing
            return json.loads(prediction_answer)
        except json.JSONDecodeError:
            pass
        # Try to extract a JSON-like block using regex
        json_pattern = re.compile(r'(\{.*?\}|\[.*?\])', re.DOTALL)
        match = json_pattern.search(prediction_answer)
        if not match:
            print('Response was not read..')
            return []
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        # Attempt special parser for mixed malformed IDs and STATUS
        if '"ID"' in json_str or '"STATUS"' in json_str:
            cleaned = self.parse_mixed_malformed_ids_and_statuses(json_str)
            if cleaned:
                return cleaned
        print('Parsing failed..')
        return []

    def safe_load_paper_items(self, data, model_class):
        """
        Safely load a list of dicts into pydantic model instances, filling missing fields with empty string.
        """
        # Safely instantiate pydantic models, filling missing fields with empty string
        return [model_class(**{k: item.get(k, "") for k in model_class.model_fields}) for item in data]

class Methods(Enum):
    """Enum for supported prediction methods."""
    TOKEN = 'token'
    ID = 'id'
    ID_TOKEN = 'id_token'



