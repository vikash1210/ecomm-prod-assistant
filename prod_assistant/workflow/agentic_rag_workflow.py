"""Agentic RAG workflow using LangChain and LangGraph-compatible APIs."""

import logging
import re
from time import perf_counter
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from prod_assistant.mcp_servers.client import MCPProductSearchClient
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between workflow nodes."""

    question: str
    search_query: str
    context: str
    answer: str


class AgenticRAG:
    """Retrieve product information and generate grounded answers."""

    FALLBACK_ANSWER = (
        "Sorry, I could not find this product or its requested "
        "details in our product catalog."
    )

    NO_CONTEXT = "No relevant product documents found."
    NO_LOCAL_RESULTS = "No local results found."

    def __init__(
        self,
        retriever=None,
        llm=None,
        max_retries: int = 1,
    ) -> None:
        """Initialize the retriever, MCP client, LLM, and retry limit."""
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        # An injected retriever is retained for tests and custom integrations.
        # Normal application execution uses the MCP product-search tool.
        self.retriever = retriever if retriever is not None else Retriever()
        self.mcp_client = (
            MCPProductSearchClient() if retriever is None else None
        )
        self.llm = llm if llm is not None else ModelLoader().load_llm()
        self.max_retries = max_retries
        self.app = self._build_workflow().compile()

        logger.info("Agentic RAG workflow initialized")

    @staticmethod
    def _is_greeting(query: str) -> bool:
        """Identify simple greetings that do not require RAG."""
        normalized = re.sub(r"[!?.\s]+", " ", query.lower()).strip()

        greetings = {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "hey there",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
        }

        return normalized in greetings

    @staticmethod
    def _format_documents(documents) -> str:
        """Format retrieved documents as LLM context."""
        if not documents:
            return AgenticRAG.NO_CONTEXT

        chunks = []

        for document in documents:
            metadata = document.metadata or {}
            content = (document.page_content or "").strip()

            title = (
                metadata.get("product_title")
                or metadata.get("title")
                or metadata.get("name")
                or "N/A"
            )
            price = metadata.get("price", "N/A")
            rating = metadata.get("rating", "N/A")

            chunks.append(
                "\n".join(
                    [
                        f"Product: {title}",
                        f"Price: {price}",
                        f"Rating: {rating}",
                        f"Details: {content}",
                    ]
                )
            )

        return "\n\n---\n\n".join(chunks)

    @staticmethod
    def _has_context(context: str) -> bool:
        """Check whether retrieval returned actual source material."""
        normalized = context.strip().casefold()
        return bool(normalized) and normalized not in {
            AgenticRAG.NO_CONTEXT.casefold(),
            AgenticRAG.NO_LOCAL_RESULTS.casefold(),
        }

    def _retrieve_context(self, query: str) -> str:
        """Retrieve product context through MCP or an injected retriever."""
        if self.mcp_client is not None:
            result = self.mcp_client.search_product(query)
            if not result or result.strip().casefold() in {
                self.NO_LOCAL_RESULTS.casefold(),
                self.NO_CONTEXT.casefold(),
            }:
                return self.NO_CONTEXT
            return result.strip()

        documents = self.retriever.invoke(query)
        return self._format_documents(documents)

    def _invoke_llm(self, template: str, **values) -> str:
        """Run a prompt through the configured LLM."""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(values).strip()

    def _is_relevant(self, question: str, context: str) -> bool:
        """Ask the LLM whether the context can answer the question."""
        if not self._has_context(context):
            return False

        response = self._invoke_llm(
            "Determine whether the supplied product context contains "
            "information relevant to answering the user's question.\n"
            "Reply with exactly YES or NO. Do not explain.\n\n"
            "User question:\n{question}\n\n"
            "Product context:\n{context}\n\n"
            "Relevance:",
            question=question,
            context=context,
        )
        return response.strip().upper().startswith("YES")

    def _rewrite_query(self, question: str, previous_query: str) -> str:
        """Generate a more effective product search query."""
        rewritten = self._invoke_llm(
            "Rewrite the user's product question as a concise search "
            "query that will help retrieve relevant product information.\n"
            "Return only the revised search query, without explanation.\n\n"
            "Original question:\n{question}\n\n"
            "Previous search query:\n{previous_query}\n\n"
            "Revised search query:",
            question=question,
            previous_query=previous_query,
        )
        return rewritten or previous_query

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate a grounded answer from product context."""
        if not self._has_context(context):
            return self.FALLBACK_ANSWER

        answer = self._invoke_llm(
            "You are ShopSmart, a helpful e-commerce product assistant.\n"
            "Answer the user's question using only the supplied product "
            "context.\n"
            "Do not invent product names, prices, currency, ratings, "
            "availability, or product details.\n"
            "If the requested information is not present, say so clearly.\n"
            "Keep the answer concise and easy to read.\n\n"
            "User question:\n{question}\n\n"
            "Retrieved product context:\n{context}\n\n"
            "Answer:",
            question=question,
            context=context,
        )
        return answer or self.FALLBACK_ANSWER

    def _assistant(self, state: AgentState) -> dict:
        """Prepare the initial search query."""
        logger.info("RAG node started: assistant")
        return {"search_query": state["question"].strip()}

    def _retrieve(self, state: AgentState) -> dict:
        """Retrieve product context using MCP or the configured retriever."""
        started_at = perf_counter()
        logger.info("RAG node started: retrieval")

        try:
            context = self._retrieve_context(state["search_query"])
            logger.info("Retrieval completed; context available: %s", self._has_context(context))
            return {"context": context}
        except Exception:
            logger.exception("Product retrieval failed")
            raise
        finally:
            logger.info(
                "Retrieval duration: %.2f seconds",
                perf_counter() - started_at,
            )

    def _generate(self, state: AgentState) -> dict:
        """Generate a grounded answer for graph-based execution."""
        started_at = perf_counter()
        logger.info("RAG node started: generation")

        try:
            answer = self._generate_answer(
                state["question"],
                state["context"],
            )
            return {"answer": answer}
        except Exception:
            logger.exception("LLM answer generation failed")
            raise
        finally:
            logger.info(
                "Generation duration: %.2f seconds",
                perf_counter() - started_at,
            )

    def _build_workflow(self):
        """Build the basic retrieval and generation graph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("assistant", self._assistant)
        workflow.add_node("retriever", self._retrieve)
        workflow.add_node("generator", self._generate)
        workflow.add_edge(START, "assistant")
        workflow.add_edge("assistant", "retriever")
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)
        return workflow

    def run(self, query: str) -> str:
        """Run retrieval, relevance checks, query rewrites, and generation."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        question = query.strip()

        if self._is_greeting(question):
            logger.info("Greeting detected; skipping RAG workflow")
            return (
                "Hi! I'm your ShopSmart product assistant. "
                "I can help you find products, compare prices, "
                "and answer questions about our catalog. "
                "What are you looking for?"
            )

        started_at = perf_counter()
        search_query = question
        context = self.NO_CONTEXT
        logger.info("Product question received")

        try:
            for attempt in range(self.max_retries + 1):
                logger.info(
                    "Retrieval attempt %d of %d",
                    attempt + 1,
                    self.max_retries + 1,
                )

                context = self._retrieve_context(search_query)

                if not self._has_context(context):
                    logger.info("No product context returned")
                elif self._is_relevant(question, context):
                    logger.info("Retrieved context judged relevant")
                    return self._generate_answer(question, context)
                else:
                    logger.info("Retrieved context judged irrelevant")

                if attempt < self.max_retries:
                    search_query = self._rewrite_query(question, search_query)
                    logger.info("Search query rewritten")
                    continue

                logger.info("Retry limit reached")
                return self._generate_answer(question, context)

            return self.FALLBACK_ANSWER

        except Exception:
            logger.exception("Agentic RAG request failed")
            raise
        finally:
            logger.info(
                "Total RAG duration: %.2f seconds",
                perf_counter() - started_at,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = AgenticRAG()
    print(agent.run("What is the price of iPhone 15?"))
