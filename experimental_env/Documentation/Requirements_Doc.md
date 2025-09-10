# APA Project Nr.8: Requirements Documentation

## Purpose

To provide a robust, modular, and extensible infrastructure for conducting medium-scale,
high-quality machine learning benchmarking experiments,
with a focus on active learning and LLM-based workflows.

## Functional Requirements

- **OOP/Modular Design:**
  - All core logic should be encapsulated in classes and modules.
  - Code should be well-documented with docstrings and comments.
- **Design Patterns:**
  - Appropriate patterns (Strategy, Factory, DAO) should be used for flexibility and clarity.
- **Integrated Preprocessing:**
  - Malformed LLM outputs and missing data should be handled gracefully.
  - Preprocessing, including class imbalance and feature vectorization, should be handled effectively.
  - Batch processing and efficient data handling should be supported.
- **Extensibility:**
  - New approaches, models, and prompt schemas should be addable with minimal changes via techniques such as inheritance.
- **Database Architecture:**
  - SQLite should be used for experiment metadata, prompt, and prediction storage.
  - Traceability and reproducibility should be ensured.
- **Linked Article Support:**
  - The system should be able to detect the effective location of the data, i.e. whether the csv file contains the data or a link to it.
  - The system should fetch and process articles from OpenAlex as well as directly from datasets.
- **LLM Communication:**
  - Multiple LLM providers should be supported.
  - Prompt/response formatting and parsing should be handled robustly.
- **Detailed Logging:**
  - Experiment metadata, technical settings, and timestamps should be logged for auditability.

## Quality Attributes

- **Traceability:**
  - Every experiment should be logged with metadata and results.
- **Usability:**
  - Clear interfaces should be provided for adding new approaches, models, and prompts.
