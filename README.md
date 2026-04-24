# LLM Evaluation Harness

A notebook-based evaluation harness for testing chatbot and Retrieval-Augmented Generation (RAG) systems with LangSmith, LangChain, and OpenAI models.

The project demonstrates how to create evaluation datasets, run an LLM or RAG application over those datasets, and grade outputs with LLM-as-a-judge evaluators for correctness, concision, relevance, groundedness, and retrieval quality.

## Project Overview

This repository contains two evaluation workflows:

- `chatbot-evaluation-pipline.ipynb`: evaluates a simple chatbot response function against reference answers.
- `rag-evaluation-pipeline.ipynb`: builds a RAG pipeline over Lilian Weng blog posts and evaluates generated answers plus retrieved context.
- `rag_strategy_evaluation.py`: compares traditional vector RAG, vectorless hierarchical RAG, and a hybrid retrieval approach in LangSmith.

## Architecture

### Chatbot Evaluation

The chatbot evaluation flow creates a LangSmith dataset, runs an application over each example, and uses an LLM judge to score the response.

<img src="Evaluation%20Architecture.png" alt="Chatbot evaluation architecture" width="900">

### RAG Evaluation

The RAG pipeline loads web documents, splits them into chunks, embeds them into an in-memory vector store, retrieves relevant chunks, and asks an LLM to answer using the retrieved context.

<img src="RAG%20Evaluation%20Architecture.png" alt="RAG evaluation architecture" width="900">

### Vectorless RAG Workflow

The vectorless RAG workflow builds a document hierarchy, scores relevant sections lexically, searches chunks inside the best sections, and preserves source/section provenance for evaluation.

<img src="Vectorless%20RAG%20Workflow.png" alt="Vectorless RAG workflow" width="900">

### Hybrid RAG Workflow

The hybrid RAG workflow combines semantic vector retrieval with hierarchy-based vectorless retrieval, deduplicates candidates, reranks them with an LLM, and answers from the best final context.

<img src="Hybrid%20RAG%20Workflow.png" alt="Hybrid RAG workflow" width="900">

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
├── rag_strategy_evaluation.py
├── Evaluation Architecture.png
├── RAG Evaluation Architecture.png
├── Vectorless RAG Workflow.png
├── Hybrid RAG Workflow.png
├── LangSmith RAG Evaluation.png
├── results/
│   ├── traditional RAG results.png
│   ├── vectorless RAG results.png
│   └── hybrid RAG results.png
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── main.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/HiteshM2107/LLM-Evaluation-Harness
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

### RAG Strategy Comparison

`rag_strategy_evaluation.py` evaluates three retrieval strategies against the same mixed dataset and evaluator set:

1. Traditional vector RAG: embeds flat chunks with `OpenAIEmbeddings` and retrieves by vector similarity.
2. Vectorless hierarchical RAG: builds a document -> section -> chunk hierarchy and retrieves with lexical section/chunk scoring, without embeddings.
3. Hybrid RAG: combines vector recall with vectorless hierarchical precision, deduplicates candidates, then reranks the merged context before answering.

The source corpus intentionally mixes narrative blog posts with structured internal-style documents, including policy, runbook, release notes, FAQ, and incident-review examples. This makes the comparison less biased toward vector-only semantic retrieval.

Run it with:

```bash
python rag_strategy_evaluation.py
```

The script creates or reuses the `RAG Strategy Comparison Mixed Corpus` LangSmith dataset, then runs separate experiments for:

- `traditional-vector-rag`
- `vectorless-hierarchical-rag`
- `hybrid-vector-vectorless-rag`

### Dataset Sources

The mixed-corpus strategy comparison uses three public Lilian Weng posts:

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)

It also includes synthetic internal-style documents defined in `rag_strategy_evaluation.py`:

- `internal://acme-support-policy`: refund rules, escalation tiers, and security exceptions
- `internal://vectorless-rag-runbook`: hierarchy indexing, retrieval behavior, and vectorless failure modes
- `internal://product-release-notes`: versioned release notes for retrieval diagnostics, hybrid mode, and reranking
- `internal://billing-faq`: billable usage, credits, and billing permissions
- `internal://incident-review-delayed-evals`: delayed evaluation job incident cause, mitigation, and prevention owner

### Example Results

The `results/` folder contains LangSmith screenshots from one mixed-corpus run. In that run, hybrid RAG produced the strongest overall result after reranking, while also showing higher latency and token usage because it performs extra LLM reranking calls.

#### Traditional Vector RAG

<img src="results/traditional%20RAG%20results.png" alt="Traditional vector RAG LangSmith results" width="900">

#### Vectorless Hierarchical RAG

<img src="results/vectorless%20RAG%20results.png" alt="Vectorless hierarchical RAG LangSmith results" width="900">

#### Hybrid Vector + Vectorless RAG

<img src="results/hybrid%20RAG%20results.png" alt="Hybrid vector and vectorless RAG LangSmith results" width="900">

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
