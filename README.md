# LLM Evaluation Harness

A notebook-based evaluation harness for testing chatbot and Retrieval-Augmented Generation (RAG) systems with LangSmith, LangChain, and OpenAI models.

The project demonstrates how to create evaluation datasets, run an LLM or RAG application over those datasets, and grade outputs with LLM-as-a-judge evaluators for correctness, concision, relevance, groundedness, and retrieval quality.

## Project Overview

This repository contains two evaluation workflows:

- `chatbot-evaluation-pipline.ipynb`: evaluates a simple chatbot response function against reference answers.
- `rag-evaluation-pipeline.ipynb`: builds a RAG pipeline over Lilian Weng blog posts and evaluates generated answers plus retrieved context.

## Architecture

### Chatbot Evaluation

The chatbot evaluation flow creates a LangSmith dataset, runs an application over each example, and uses an LLM judge to score the response.

<img src="Evaluation%20Architecture.png" alt="Chatbot evaluation architecture" width="900">

### RAG Evaluation

The RAG pipeline loads web documents, splits them into chunks, embeds them into an in-memory vector store, retrieves relevant chunks, and asks an LLM to answer using the retrieved context.

<img src="RAG%20Evaluation%20Architecture.png" alt="RAG evaluation architecture" width="900">

### RAG Evaluation Flow

This diagram shows how the RAG pipeline is evaluated in LangSmith using multiple evaluators such as correctness, relevance, groundedness, and retrieval relevance.

<img src="LangSmith%20RAG%20Evaluation.png" alt="RAG evaluation flow in LangSmith" width="900">

## Features

- Create LangSmith datasets programmatically.
- Trace chatbot and RAG functions with `@traceable()`.
- Evaluate model responses with custom LLM-as-a-judge metrics.
- Compare answer correctness against reference outputs.
- Check response relevance without requiring ground truth answers.
- Check groundedness against retrieved source documents.
- Check retrieval relevance between the user question and retrieved chunks.
- Export LangSmith experiment results to a local pandas dataframe.

## Tech Stack

- Python 3.12+
- LangChain
- LangSmith
- OpenAI
- In-memory LangChain vector stores
- Jupyter
- pandas
- python-dotenv

## Repository Structure

```text
.
├── chatbot-evaluation-pipline.ipynb
├── rag-evaluation-pipeline.ipynb
├── Evaluation Architecture.png
├── RAG Evaluation Architecture.png
├── LangSmith RAG Evaluation.png
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── main.py
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "LLM Evaluation Harness"
```

### 2. Create the environment

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

Do not commit your `.env` file to GitHub.

### 4. Open Jupyter

```bash
jupyter notebook
```

Then open either notebook:

- `chatbot-evaluation-pipline.ipynb`
- `rag-evaluation-pipeline.ipynb`

If you are using VS Code, select the `LLM Evaluation Harness` notebook kernel.

## Notebooks

### Chatbot Evaluation Pipeline

This notebook shows how to:

1. Load environment variables.
2. Create a LangSmith dataset.
3. Define a simple chatbot application.
4. Define evaluator functions such as correctness and concision.
5. Run evaluation experiments in LangSmith.
6. Compare results across model versions.

If you rerun the dataset creation cell without changing the dataset name, LangSmith may raise a conflict error because the dataset already exists.

### RAG Evaluation Pipeline

This notebook shows how to:

1. Load documents from public URLs.
2. Split documents into chunks.
3. Create an in-memory vector store with OpenAI embeddings.
4. Retrieve relevant documents for a question.
5. Build a traced RAG function with `@traceable()`.
6. Create a RAG test dataset in LangSmith.
7. Evaluate the pipeline with:
   - Correctness
   - Relevance
   - Groundedness
   - Retrieval relevance
8. Convert experiment results to a pandas dataframe.

## Evaluation Metrics

### Correctness

Compares the model answer against a reference answer and determines whether the generated response is factually correct.

### Relevance

Checks whether the generated answer directly addresses the user question and is useful.

### Groundedness

Checks whether the generated answer is supported by the retrieved source documents.

### Retrieval Relevance

Checks whether the retrieved documents are relevant to the input question.

### Concision

Checks whether a chatbot answer is reasonably concise compared with the expected answer.

## Notes

- Running the notebooks will make OpenAI API calls.
- LangSmith experiment results are visible in your LangSmith workspace.
- Rerunning dataset creation cells may fail unless you reuse an existing LangSmith dataset or change the dataset name.
- The RAG notebook uses public Lilian Weng blog posts as the source corpus.
- The project is designed for learning and experimentation rather than production deployment.

## Future Improvements

- Add automated scripts for running evaluations outside notebooks.
- Add more datasets and larger test sets.
- Add model comparison reports.
- Persist vector stores instead of rebuilding them in-memory.
- Add CI checks for notebook execution.
