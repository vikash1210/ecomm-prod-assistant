"""Tests for the agentic RAG workflow."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from prod_assistant.workflow.agentic_rag_workflow import AgenticRAG


class FakeRetriever:
    """Return predictable documents and record search queries."""

    def __init__(self, results=None):
        self.results = results or []
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.results


def make_fake_llm(responses):
    """Create a LangChain-compatible fake model."""
    iterator = iter(responses)

    def respond(_prompt):
        return AIMessage(content=next(iterator))

    return RunnableLambda(respond)


def test_workflow_generates_answer_when_context_is_relevant():
    retriever = FakeRetriever(
        [
            SimpleNamespace(
                page_content="iPhone 15 price is 50000 INR.",
                metadata={"product_title": "iPhone 15", "price": 50000},
            )
        ]
    )
    llm = make_fake_llm(
        ["YES", "The iPhone 15 costs INR 50,000."]
    )
    agent = AgenticRAG(retriever=retriever, llm=llm)

    answer = agent.run("What is the price of iPhone 15?")

    assert "50,000" in answer
    assert len(retriever.queries) == 1


def test_workflow_rewrites_query_when_context_is_irrelevant():
    retriever = FakeRetriever(
        [
            SimpleNamespace(
                page_content="Product details.",
                metadata={"product_title": "Phone", "price": 10000},
            )
        ]
    )
    llm = make_fake_llm(
        [
            "NO",
            "iPhone 15 current price",
            "YES",
            "The iPhone 15 price is not specified.",
        ]
    )
    agent = AgenticRAG(retriever=retriever, llm=llm, max_retries=1)

    answer = agent.run("Tell me about iPhone 15.")

    assert answer
    assert len(retriever.queries) == 2
    assert retriever.queries[1] == "iPhone 15 current price"


def test_workflow_stops_rewriting_after_retry_limit():
    retriever = FakeRetriever([])
    llm = make_fake_llm(
        [
            "NO",
            "revised product search",
            "NO",
            "No product information is available in the context.",
        ]
    )
    agent = AgenticRAG(retriever=retriever, llm=llm, max_retries=1)

    answer = agent.run("Find a product.")

    assert answer
    assert len(retriever.queries) == 2


def test_workflow_rejects_empty_query():
    agent = AgenticRAG(
        retriever=FakeRetriever([]),
        llm=make_fake_llm([]),
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        agent.run("   ")


def test_workflow_rejects_negative_retry_limit():
    with pytest.raises(ValueError, match="max_retries cannot be negative"):
        AgenticRAG(
            retriever=FakeRetriever([]),
            llm=make_fake_llm([]),
            max_retries=-1,
        )