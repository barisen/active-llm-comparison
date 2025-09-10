# APA Project Nr.8 Architecture

## Overview

This project provides a robust, modular infrastructure for conducting medium-scale, high-quality machine learning experiments for active learning and LLM-based workflows. The design is OOP, and follows modular architecture with extensibility and maintainability as core goals.

## Main Components

### Experiment

- **Role:** Orchestrates the entire experiment lifecycle: data loading, preprocessing, batch management, and execution.
- **Responsibilities:**
  - Loads and preprocesses datasets (including OpenAlex integration).
  - Manages initial data splits (positives/negatives).
  - Configures batch settings for training and inference.
  - Runs the experiment using a selected approach and model.
  - Handles saving of results and metadata to the database.

### Approach

- **Role:** Encapsulates experiment strategies (e.g., Active Learning, Low-Shot Learning).
- **Responsibilities:**
  - Splits data into batches based on user settings.
  - Processes each batch to keep data and labels aligned.
  - Dynamically moves test data to training in active learning.
  - Handles vectorization and preprocessing (e.g., TF-IDF for classical models).
  - Supports extensibility for new strategies via `BaseApproach`.

### Model

- **Role:** Abstracts both classical ML models and LLMs.
- **Responsibilities:**
  - Provides a unified interface for training and prediction.
  - Supports classical models (Naive Bayes, Random Forest, Logistic Regression) and LLMs (HuggingFace, Gemini, custom endpoints).
  - Handles LLM communication if needed.
  - Easily extendable for new model types/providers.
  - Adapts workflow based on the chosen approach (e.g., single vs. multiple data points per iteration).

### Prompt

- **Role:** Handles prompt creation, schema validation, and robust parsing for LLM-based experiments.
- **Responsibilities:**
  - Generates augmentation and prediction prompts from user prompt templates.
  - Parses and validates LLM responses, including malformed outputs via JSON, Regex, etc.
  - Supports multiple response schemas.

### DAO

- **Role:** Data Access Object for experiment persistence.
- **Responsibilities:**
  - Saves experiment metadata, prompts, and predictions to a SQLite database.
  - Ensures experiment traceability and reproducibility.

## Data Flow

1. **Dataset Loading:** `Experiment` loads and preprocesses data.
2. **Approach Selection:** User selects an `Approach` (e.g., Active, Low-Shot).
3. **Model Selection:** User selects a `Model` (classical or LLM).
4. **Prompting (if LLM):** `Prompt` generates and parses LLM prompts/responses.
5. **Batch Execution:** `Approach` manages batch logic and calls the `Model`.
6. **Results Saving:** `DAO` persists results and metadata.

## Extensibility

- New approaches: Subclass `BaseApproach`.
- New models: Subclass `APABaseModel` or `BaseLLM`.
- New prompt schemas: Extend `Prompt`.
- Database schema can be extended for new metadata.

## Design Patterns Used

- **Strategy:** For experiment approaches.
- **Factory:** For model instantiation.
- **DAO:** For persistence.
- **OOP/Modular:** For modularity and extensibility.

## Logging & Quality

- Detailed logging of all experiment metadata and timestamps is supported.
- Database ensures traceability and reproducibility.
