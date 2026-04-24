"""Compare vector, vectorless, and hybrid RAG strategies with LangSmith.

This script mirrors the existing notebook evaluation flow, but makes the
retrieval strategy explicit so experiments can be compared side by side.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

# Keep optional ML stacks quiet when LangChain imports text splitter modules in environments that also have TensorFlow/Keras installed.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("USER_AGENT", "llm-evaluation-harness/0.1")

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict


URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

RAG_DATASET_NAME = "RAG Strategy Comparison Mixed Corpus"

MIXED_CORPUS_DOCS = [
    Document(
        page_content="""# Acme Support Policy

## Refunds
Standard subscription refunds are available within 14 calendar days of purchase when usage is below 100 API calls.
Enterprise contract refunds require written approval from Finance and Customer Success.
Refund requests must include the workspace ID, invoice ID, and requester email.

## Escalation Tiers
Tier 1 issues are general product questions and should receive a first response within 24 business hours.
Tier 2 issues include degraded performance, failed scheduled jobs, and billing discrepancies.
Tier 3 issues include production outages, data loss risk, security incidents, and executive escalations.
Tier 3 issues require paging the on-call engineer and posting an update every 30 minutes until mitigation.

## Security Exceptions
Security exceptions expire after 30 days unless renewed by the security owner.
Temporary allowlist requests must include a business justification, IP range, owner, and expiration date.
Production secrets must never be pasted into support tickets.
""",
        metadata={
            "source": "internal://acme-support-policy",
            "title": "Acme Support Policy",
            "document_type": "policy",
        },
    ),
    Document(
        page_content="""# Vectorless RAG Runbook

## Indexing
The vectorless retriever stores documents as a hierarchy of document, section, and chunk nodes.
Each node keeps a stable path, heading, source URI, and parent identifier.
Chunks are selected only after the retriever has identified likely parent sections.

## Retrieval
For exact lookup questions, first match headings, IDs, and domain terms against the hierarchy.
For semantic questions, use broader lexical expansion and inspect sibling chunks under the matched section.
For cross-document questions, retrieve candidate sections from multiple documents before selecting final chunks.

## Failure Modes
Vectorless retrieval can fail when the query uses synonyms that do not appear in headings or source text.
It can also over-prioritize a heading match when the body text does not answer the question.
Hybrid retrieval should add semantic candidates to recover from synonym mismatch.
""",
        metadata={
            "source": "internal://vectorless-rag-runbook",
            "title": "Vectorless RAG Runbook",
            "document_type": "runbook",
        },
    ),
    Document(
        page_content="""# Product Release Notes

## Version 2.1.0
Release date: 2026-02-14
Feature: Added hierarchy-aware retrieval diagnostics.
Metric added: parent_section_hit_rate.
Known limitation: diagnostics do not include reranker explanations.

## Version 2.2.0
Release date: 2026-03-28
Feature: Added hybrid retrieval mode.
Metric added: hybrid_context_overlap.
Known limitation: hybrid mode may add noisy context if semantic and hierarchy results disagree.

## Version 2.3.0
Release date: 2026-04-12
Feature: Added reranker preview for merged retrieval candidates.
Metric added: reranked_context_precision.
Known limitation: reranker preview increases evaluation cost.
""",
        metadata={
            "source": "internal://product-release-notes",
            "title": "Product Release Notes",
            "document_type": "release_notes",
        },
    ),
    Document(
        page_content="""# Billing FAQ

## What counts as billable usage?
Billable usage includes successful model calls, embedding requests, and batch evaluation runs.
Failed authentication requests are not billable.
Cancelled evaluation runs are billed only for completed model calls.

## How do credits apply?
Credits are applied before monthly invoices are finalized.
Promotional credits expire 90 days after issue.
Enterprise credits follow the expiration date written in the order form.

## Who can change billing settings?
Only workspace admins and organization owners can change billing settings.
Support agents can view invoice status but cannot change payment methods.
""",
        metadata={
            "source": "internal://billing-faq",
            "title": "Billing FAQ",
            "document_type": "faq",
        },
    ),
    Document(
        page_content="""# Incident Review: Delayed Evaluation Jobs

On March 4, evaluation jobs in the east region were delayed for 47 minutes.
The immediate cause was a queue worker deployment that reduced concurrency from 20 workers to 6 workers.
The customer-visible symptom was that experiment results appeared late even though traces were still being collected.
The mitigation was to roll back the worker deployment and replay queued jobs.

The prevention item was to add a deployment check that compares worker concurrency against the previous stable release.
The owner for the prevention item is Platform Reliability.
The target completion date is April 30, 2026.
""",
        metadata={
            "source": "internal://incident-review-delayed-evals",
            "title": "Incident Review: Delayed Evaluation Jobs",
            "document_type": "incident_review",
        },
    ),
]

RAG_EXAMPLES = [
    {
        "inputs": {"question": "How does the ReAct agent use self-reflection?"},
        "outputs": {
            "answer": (
                "ReAct integrates reasoning and acting by using actions such as "
                "Wikipedia search, observing tool outputs, and reasoning over those "
                "observations to update its next steps."
            )
        },
    },
    {
        "inputs": {"question": "What are the types of biases that can arise with few-shot prompting?"},
        "outputs": {
            "answer": (
                "The biases that can arise with few-shot prompting include majority "
                "label bias, recency bias, and common token bias."
            )
        },
    },
    {
        "inputs": {"question": "What are five types of adversarial attacks?"},
        "outputs": {
            "answer": (
                "Five adversarial attack types include token manipulation, "
                "gradient-based attacks, jailbreak prompting, human red-teaming, "
                "and model red-teaming."
            )
        },
    },
    {
        "inputs": {"question": "How are planning, memory, and tool use related in LLM agents?"},
        "outputs": {
            "answer": (
                "Planning lets an agent decompose and refine tasks, memory stores "
                "short- and long-term context, and tool use lets the agent call "
                "external resources to complete tasks."
            )
        },
    },
    {
        "inputs": {"question": "Compare chain-of-thought prompting and tree-of-thought prompting."},
        "outputs": {
            "answer": (
                "Chain-of-thought prompting asks the model to reason through a linear "
                "sequence of steps, while tree-of-thought prompting explores multiple "
                "reasoning paths and can evaluate or backtrack among them."
            )
        },
    },
    {
        "inputs": {"question": "What conditions must be met for a standard subscription refund?"},
        "outputs": {
            "answer": (
                "A standard subscription refund is available within 14 calendar days "
                "of purchase when usage is below 100 API calls."
            )
        },
    },
    {
        "inputs": {"question": "Which release added hybrid retrieval mode and what limitation did it have?"},
        "outputs": {
            "answer": (
                "Version 2.2.0 added hybrid retrieval mode, and its known limitation "
                "was that it may add noisy context when semantic and hierarchy results disagree."
            )
        },
    },
    {
        "inputs": {"question": "How should hybrid retrieval recover when vectorless retrieval has synonym mismatch?"},
        "outputs": {
            "answer": (
                "Hybrid retrieval should add semantic candidates so it can recover "
                "when vectorless retrieval misses relevant content due to synonym mismatch."
            )
        },
    },
    {
        "inputs": {"question": "Are failed authentication requests billable usage?"},
        "outputs": {"answer": "No. Failed authentication requests are not billable."},
    },
    {
        "inputs": {"question": "What caused delayed evaluation jobs in the east region and who owns the prevention item?"},
        "outputs": {
            "answer": (
                "The delay was caused by a queue worker deployment that reduced "
                "concurrency from 20 workers to 6 workers, and Platform Reliability "
                "owns the prevention item."
            )
        },
    },
]


def load_settings() -> None:
    load_dotenv(dotenv_path=".env", override=True)
    os.environ.setdefault("USER_AGENT", "llm-evaluation-harness/0.1")
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")


def load_source_documents(urls: list[str] = URLS) -> list[Document]:
    docs_by_url = [WebBaseLoader(url).load() for url in urls]
    web_docs = [doc for docs in docs_by_url for doc in docs]
    return web_docs + MIXED_CORPUS_DOCS


def make_flat_chunks(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": f"flat-{index}",
            "retrieval_level": "chunk",
        }
    return chunks


def make_vector_retriever(chunks: list[Document], k: int = 6):
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


@dataclass(frozen=True)
class HierarchicalIndex:
    documents: list[Document]
    sections: list[Document]
    chunks_by_section: dict[str, list[Document]]


HEADER_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,:;()/&.'-]{2,90}#?$")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


def parse_sections(doc: Document, document_index: int) -> list[Document]:
    title = doc.metadata.get("title") or doc.metadata.get("source") or f"Document {document_index}"
    lines = [line.strip() for line in doc.page_content.splitlines()]
    sections: list[tuple[str, list[str]]] = []
    current_title = str(title)
    current_lines: list[str] = []

    for line in lines:
        if not line:
            continue

        markdown_heading = line.startswith("#")
        normalized = line.strip("#").strip() if markdown_heading else line.rstrip("#").strip()
        is_heading = markdown_heading or line.endswith("#") or (
            len(normalized.split()) <= 10
            and HEADER_RE.match(line) is not None
            and len(current_lines) >= 4
        )
        if is_heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = normalized
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    section_docs = []
    for section_index, (section_title, section_lines) in enumerate(sections):
        section_id = f"doc-{document_index}-section-{section_index}"
        section_docs.append(
            Document(
                page_content="\n".join(section_lines),
                metadata={
                    **doc.metadata,
                    "document_id": f"doc-{document_index}",
                    "section_id": section_id,
                    "section": section_title,
                    "section_index": section_index,
                    "retrieval_level": "section",
                },
            )
        )
    return section_docs


def make_hierarchical_index(docs: list[Document]) -> HierarchicalIndex:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    sections: list[Document] = []
    chunks_by_section: dict[str, list[Document]] = defaultdict(list)

    for document_index, doc in enumerate(docs):
        sections.extend(parse_sections(doc, document_index))

    for section in sections:
        section_chunks = splitter.split_documents([section])
        section_id = section.metadata["section_id"]
        for chunk_index, chunk in enumerate(section_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_id": f"{section_id}-chunk-{chunk_index}",
                "chunk_index": chunk_index,
                "retrieval_level": "chunk",
            }
            chunks_by_section[section_id].append(chunk)

    return HierarchicalIndex(
        documents=docs,
        sections=sections,
        chunks_by_section=dict(chunks_by_section),
    )


def lexical_score(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    text_counts = Counter(tokenize(text))
    overlap = sum(min(query_counts[token], text_counts[token]) for token in query_counts)
    coverage = overlap / max(len(query_counts), 1)
    density = overlap / max(sum(text_counts.values()), 1)
    phrase_bonus = 0.25 if " ".join(query_tokens[:2]) in text.lower() else 0.0
    return coverage + density + phrase_bonus


class VectorlessHierarchicalRetriever:
    """Hierarchy-first retrieval without embeddings.

    It first scores document sections lexically, then searches chunks only within
    the best sections. This approximates "precise retrieval" without a vector
    index and keeps source provenance in metadata.
    """

    def __init__(self, index: HierarchicalIndex, section_k: int = 4, chunk_k: int = 6):
        self.index = index
        self.section_k = section_k
        self.chunk_k = chunk_k

    def invoke(self, question: str) -> list[Document]:
        query_tokens = tokenize(question)
        ranked_sections = sorted(
            self.index.sections,
            key=lambda section: lexical_score(
                query_tokens,
                f"{section.metadata.get('section', '')}\n{section.page_content}",
            ),
            reverse=True,
        )[: self.section_k]

        candidate_chunks = [
            chunk
            for section in ranked_sections
            for chunk in self.index.chunks_by_section.get(section.metadata["section_id"], [])
        ]
        ranked_chunks = sorted(
            candidate_chunks,
            key=lambda chunk: lexical_score(
                query_tokens,
                f"{chunk.metadata.get('section', '')}\n{chunk.page_content}",
            ),
            reverse=True,
        )[: self.chunk_k]

        for rank, chunk in enumerate(ranked_chunks, start=1):
            chunk.metadata = {
                **chunk.metadata,
                "retrieval_strategy": "vectorless_hierarchy",
                "retrieval_rank": rank,
            }
        return ranked_chunks


class HybridRetriever:
    """Combines vector retrieval recall with vectorless hierarchical precision."""

    def __init__(
        self,
        vector_retriever,
        vectorless_retriever: VectorlessHierarchicalRetriever,
        reranker: "HybridReranker",
        k: int = 6,
        candidate_k: int = 12,
    ):
        self.vector_retriever = vector_retriever
        self.vectorless_retriever = vectorless_retriever
        self.reranker = reranker
        self.k = k
        self.candidate_k = candidate_k

    def invoke(self, question: str) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(question)
        vectorless_docs = self.vectorless_retriever.invoke(question)
        merged: dict[str, Document] = {}

        for rank, doc in enumerate(vectorless_docs, start=1):
            key = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source')}:{doc.page_content[:80]}"
            merged[key] = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "retrieval_strategy": "hybrid",
                    "hybrid_sources": ["vectorless"],
                    "vectorless_rank": rank,
                },
            )

        for rank, doc in enumerate(vector_docs, start=1):
            key = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source')}:{doc.page_content[:80]}"
            if key in merged:
                merged[key].metadata["hybrid_sources"].append("vector")
                merged[key].metadata["vector_rank"] = rank
            else:
                merged[key] = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "retrieval_strategy": "hybrid",
                        "hybrid_sources": ["vector"],
                        "vector_rank": rank,
                    },
                )

        def hybrid_sort_key(doc: Document) -> tuple[int, int]:
            sources = doc.metadata.get("hybrid_sources", [])
            source_bonus = 0 if len(sources) > 1 else 1
            best_rank = min(doc.metadata.get("vectorless_rank", 99), doc.metadata.get("vector_rank", 99))
            return source_bonus, best_rank

        candidates = sorted(merged.values(), key=hybrid_sort_key)[: self.candidate_k]
        reranked = []
        for doc in candidates:
            score = self.reranker.score(question, doc)
            agreement_bonus = 1 if len(doc.metadata.get("hybrid_sources", [])) > 1 else 0
            doc.metadata = {
                **doc.metadata,
                "rerank_score": score,
                "agreement_bonus": agreement_bonus,
            }
            reranked.append(doc)

        return sorted(
            reranked,
            key=lambda doc: (
                doc.metadata.get("rerank_score", 0) + doc.metadata.get("agreement_bonus", 0),
                -min(doc.metadata.get("vectorless_rank", 99), doc.metadata.get("vector_rank", 99)),
            ),
            reverse=True,
        )[: self.k]


def format_documents(docs: Iterable[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        section = doc.metadata.get("section")
        heading = f"Source: {source}"
        if section:
            heading += f"\nSection: {section}"
        formatted.append(f"{heading}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def answer_with_retriever(question: str, retriever, llm) -> dict:
    docs = retriever.invoke(question)
    docs_string = format_documents(docs)
    instructions = f"""You are a helpful assistant who answers using source documents.
Use only the documents below to answer the user's question.
If the documents do not contain the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
    ai_msg = llm.invoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ]
    )
    return {"answer": ai_msg.content, "documents": docs}


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is grounded in the documents"]


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant"]


class RerankGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain why this document is or is not useful"]
    relevance: Annotated[
        Literal["high", "medium", "low"],
        ...,
        "How useful the document is for answering the question",
    ]
    score: Annotated[int, ..., "A relevance score from 0 to 10"]


CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH ANSWER, and the STUDENT ANSWER.
Grade the student answer based only on factual accuracy relative to the ground truth answer.
The student answer may contain extra information if it is factually accurate and non-conflicting.
Return correct=True only when the answer satisfies these criteria."""

RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.
Return relevant=True only when the student answer directly and usefully addresses the question."""

GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.
Return grounded=True only when the student answer is supported by the facts and does not hallucinate."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading retrieval quality.

You will be given a QUESTION and retrieved FACTS.
Return relevant=True when the facts contain keywords or semantic meaning related to the question.
Return relevant=False when the facts are completely unrelated."""

RERANK_INSTRUCTIONS = """You are ranking retrieved context for a RAG system.

You will be given a QUESTION and one candidate DOCUMENT.
Score the document by how directly it helps answer the question.

Use this rubric:
- 9-10: directly answers the question with specific facts
- 6-8: partially answers the question or gives important supporting context
- 3-5: topically related but unlikely to help answer precisely
- 0-2: unrelated or misleading

Prefer documents with exact section, policy, version, metric, date, owner, or procedural details when the question asks for precise facts."""


class HybridReranker:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0).with_structured_output(
            RerankGrade,
            method="json_schema",
            strict=True,
        )

    def score(self, question: str, doc: Document) -> int:
        candidate = format_documents([doc])
        grade = self.llm.invoke(
            [
                {"role": "system", "content": RERANK_INSTRUCTIONS},
                {"role": "user", "content": f"QUESTION: {question}\n\nDOCUMENT:\n{candidate}"},
            ]
        )
        return max(0, min(10, int(grade["score"])))


def make_evaluators() -> list[Callable]:
    correctness_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        CorrectnessGrade,
        method="json_schema",
        strict=True,
    )
    relevance_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        RelevanceGrade,
        method="json_schema",
        strict=True,
    )
    grounded_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        GroundedGrade,
        method="json_schema",
        strict=True,
    )
    retrieval_relevance_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        RetrievalRelevanceGrade,
        method="json_schema",
        strict=True,
    )

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        answers = f"""QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
        grade = correctness_llm.invoke(
            [
                {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
                {"role": "user", "content": answers},
            ]
        )
        return grade["correct"]

    def relevance(inputs: dict, outputs: dict) -> bool:
        answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = relevance_llm.invoke(
            [
                {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["relevant"]

    def groundedness(inputs: dict, outputs: dict) -> bool:
        facts = format_documents(outputs["documents"])
        answer = f"FACTS: {facts}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grounded_llm.invoke(
            [
                {"role": "system", "content": GROUNDED_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["grounded"]

    def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
        facts = format_documents(outputs["documents"])
        answer = f"QUESTION: {inputs['question']}\nFACTS: {facts}"
        grade = retrieval_relevance_llm.invoke(
            [
                {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["relevant"]

    return [correctness, groundedness, relevance, retrieval_relevance]


def get_or_create_dataset(client: Client, dataset_name: str = RAG_DATASET_NAME) -> str:
    try:
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=dataset.id, examples=RAG_EXAMPLES)
    except Exception as exc:
        if "already exists" not in str(exc).lower() and "conflict" not in str(exc).lower():
            raise
    return dataset_name


def build_retrievers() -> tuple[object, VectorlessHierarchicalRetriever, HybridRetriever]:
    source_docs = load_source_documents()
    flat_chunks = make_flat_chunks(source_docs)
    vector_retriever = make_vector_retriever(flat_chunks, k=6)
    hierarchical_index = make_hierarchical_index(source_docs)
    vectorless_retriever = VectorlessHierarchicalRetriever(hierarchical_index, section_k=4, chunk_k=6)
    hybrid_retriever = HybridRetriever(
        vector_retriever,
        vectorless_retriever,
        reranker=HybridReranker(),
        k=6,
        candidate_k=12,
    )
    return vector_retriever, vectorless_retriever, hybrid_retriever


def make_targets() -> dict[str, Callable[[dict], dict]]:
    llm = init_chat_model("openai:gpt-4o-mini")
    vector_retriever, vectorless_retriever, hybrid_retriever = build_retrievers()

    @traceable(name="traditional-vector-rag")
    def traditional_vector_rag(question: str) -> dict:
        return answer_with_retriever(question, vector_retriever, llm)

    @traceable(name="vectorless-hierarchical-rag")
    def vectorless_rag(question: str) -> dict:
        return answer_with_retriever(question, vectorless_retriever, llm)

    @traceable(name="hybrid-vector-vectorless-rag")
    def hybrid_rag(question: str) -> dict:
        return answer_with_retriever(question, hybrid_retriever, llm)

    return {
        "traditional-vector-rag": lambda inputs: traditional_vector_rag(inputs["question"]),
        "vectorless-hierarchical-rag": lambda inputs: vectorless_rag(inputs["question"]),
        "hybrid-vector-vectorless-rag": lambda inputs: hybrid_rag(inputs["question"]),
    }


def run_strategy_evaluations() -> dict[str, object]:
    load_settings()
    client = Client()
    dataset_name = get_or_create_dataset(client)
    evaluators = make_evaluators()
    targets = make_targets()

    results = {}
    for strategy_name, target in targets.items():
        results[strategy_name] = client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=strategy_name,
            metadata={
                "retrieval_strategy": strategy_name,
                "dataset": dataset_name,
            },
        )
    return results


if __name__ == "__main__":
    experiment_results = run_strategy_evaluations()
    for strategy_name, result in experiment_results.items():
        print(f"\n{strategy_name}")
        try:
            print(result.to_pandas())
        except Exception:
            print(result)
