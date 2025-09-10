

from sklearn.naive_bayes import MultinomialNB
from gradio_client import Client
import numpy as np
import contextlib
import io
from huggingface_hub import InferenceClient
import requests
from google import genai
from google.genai import types
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from abc import ABC, abstractmethod


class APABaseModel(ABC):
    """
    Abstract base class for all models in the tool.
    Enforces the interface for training and prediction.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict_next(self, novel_data):
        """Return the record_id of the most likely positive instance from novel_data."""
        pass

    @abstractmethod
    def predict(self, novel_data):
        """Return predictions for all instances in novel_data."""
        pass




class BaseLLM(APABaseModel):
    """
    Abstract base class for LLM-based models.
    Defines the interface for LLM training and prediction.
    """
    @abstractmethod
    def train(self, training_X, training_y):
        """Train the LLM (if supported)."""
        pass

    @abstractmethod
    def predict_next(self, training_X, training_y):
        """Return the record_id of the most likely positive instance (LLM-based)."""
        pass

    @abstractmethod
    def predict(self, training_X, training_y):
        """Return predictions for all instances (LLM-based)."""
        pass

    def predict_next(self, novel_data):
        """
        Default implementation: select the first positive prediction from the LLM output.
        Returns None if no positive found.
        """
        # Get predictions from the LLM
        temp_predictions = self.predict(novel_data)
        # Find the first row with a positive prediction
        positive_row = temp_predictions[temp_predictions['prediction'] == 1].head(1)
        if not positive_row.empty:
            return str(positive_row['record_id'].iloc[0])
        else:
            return None
        
    


class BaseHuLLM(BaseLLM):
    def __init__(self,client_url,prompt):
        self._prompt=prompt
        # Silence the initialization message from the client
        with contextlib.redirect_stdout(io.StringIO()):
            self.client=Client(client_url)

    def train(self, training_X,training_y):
        # If augmentation is defined, create and send augmentation prompt
        if(self._prompt.augmentation!=None):
            PROMPT=self._prompt.create_augmentation_prompt(training_X,training_y)
            self.client.predict(PROMPT,api_name="/chat")

    def predict(self, novel_data):
        # Create prediction prompt and get result from client
        PROMPT=self._prompt.create_prediction_prompt(novel_data,add_format=True)
        result=self.client.predict(PROMPT,api_name="/chat")
        return self._prompt.read_predictions(result,novel_data)
    
    
class HuLLM1(BaseHuLLM):
    def __init__(self,prompt):
        super().__init__("https://llm1-compute.cms.hu-berlin.de/",prompt)



class HuLLM3(BaseHuLLM):
    def __init__(self,prompt):
        super().__init__("https://llm3-compute.cms.hu-berlin.de/",prompt)


class HuggingFaceLLM(BaseLLM):

    def __init__(self,model_name,api_key,prompt):
        self._prompt=prompt
        # Parse provider and model name from string (format: [PROVIDER]/[MODEL])
        provider=model_name[:model_name.index('/')]
        self._model_name=model_name[model_name.index('/')+1:]
        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(
            provider=provider,
            api_key=api_key,
        )
        # Prepare schema properties for prompt parsing (if needed)
        schema_properties = prompt.response_schema.schema()['properties']
        filtered_properties = {
            field: {k: v for k, v in props.items() if k != 'title'}
            for field, props in schema_properties.items()
        }

  
        

    def train(self, training_X,training_y):
        # If augmentation is defined, create augmentation prompt
        if(self._prompt.augmentation!=None):
            self.augmentation_prompt=self._prompt.create_augmentation_prompt(training_X,training_y)

        '''
        PROMPT=self._prompt.create_prediction_prompt(novel_data)
        result=self.client.predict(PROMPT,api_name="/chat")
        return self._prompt.read_next_prediction(result)
        '''


    def predict(self, novel_data):
        # Create prediction prompt and send to HuggingFace client
        PROMPT=self._prompt.create_prediction_prompt(novel_data,add_format=True)
        stream = self.client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "user", "content": self.augmentation_prompt},
                {"role": "user", "content": PROMPT}],temperature=0.3,top_p=0.7,stream=True,
        )
        result = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        return self._prompt.read_predictions(result,novel_data)
    


class Gemini(BaseLLM):
   

    def __init__(self,model_name,api_key,prompt):
        self._prompt=prompt
        # Initialize Gemini client
        self.client = genai.Client(
            api_key=api_key,
        )
        self._model_name=model_name

    
    def train(self, training_X,training_y):
        # If augmentation is defined, create augmentation prompt
        if(self._prompt.augmentation!=None):
            self.augmentation_prompt=self._prompt.create_augmentation_prompt(training_X,training_y)



    def predict(self, novel_data):
        # Create prediction prompt and send to Gemini client
        PROMPT=self._prompt.create_prediction_prompt(novel_data)
        result = ""
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=self.augmentation_prompt),
                    types.Part.from_text(text=PROMPT)
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=list[self._prompt.response_schema],
        )
        for chunk in self.client.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        return self._prompt.read_predictions(result,novel_data)


class OpenWebUILLM(BaseLLM):

    def __init__(self,model_name,api_key,prompt):
        self._prompt=prompt
        # Parse provider and model name from string (format: [PROVIDER]/[MODEL])
        provider=model_name[:model_name.index('/')]
        self._model_name=model_name[model_name.index('/')+1:]
        # Set up local WebUI API URL
        self.url = 'http://localhost:3333/api/chat/completions'
        

  
        

    def train(self, training_X,training_y):
        # If augmentation is defined, create augmentation prompt
        if(self._prompt.augmentation!=None):
            self.augmentation_prompt=self._prompt.create_augmentation_prompt(training_X,training_y)

       


    def predict(self, novel_data):
        # Create prediction prompt and send to local WebUI API
        PROMPT=self._prompt.create_prediction_prompt(novel_data,add_format=True)
        data = {
            "model": self._model_name,
            "messages": [
                {"role": "user", "content": self.augmentation_prompt},
                {"role": "user", "content": PROMPT}
            ]
        }
        # Send request to local WebUI API (commented out alternative)
        #response = requests.post(url=self.url, headers=self.headers, json=data)
        response = requests.post(
            f"{self.url}/api/generate",
            json={"model": "llama3", "prompt": PROMPT}
        )
        # Parse response and print result
        result=response.json()["response"]
        return self._prompt.read_predictions(result,novel_data)

    
class ModelConverter:
    @staticmethod
    def convert(model_str,api_key=None,prompt=None):
        # Convert model string to the appropriate model class
        model_str_lower=model_str.lower()
        if 'hu' in model_str_lower and '1' in model_str_lower:
            return HuLLM1(prompt)
        elif 'hu' in model_str_lower and '3' in model_str_lower:
            return HuLLM3(prompt)
        elif 'gemini' in model_str_lower:
            return Gemini(model_str,api_key,prompt)
        elif 'webui' in model_str_lower:
            return OpenWebUILLM(model_str,api_key,prompt)
        else:
            return HuggingFaceLLM(model_str,api_key,prompt)

