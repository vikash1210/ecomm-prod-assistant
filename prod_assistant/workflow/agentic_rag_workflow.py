
"""Fast Agentic RAG workflow using LangGraph."""

import logging
import re
from time import perf_counter
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

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

    def __init__(self, retriever=None, llm=None) -> None:
        """Initialize the retriever, LLM, and workflow."""

        self.retriever = retriever if retriever is not None else Retriever()
        self.llm = llm if llm is not None else ModelLoader().load_llm()

        self.app = self._build_workflow().compile()

        logger.info("Fast Agentic RAG workflow initialized")

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
        """Check whether retrieval returned useful source material."""

        return bool(context.strip()) and (
            context.strip() != AgenticRAG.NO_CONTEXT
        )

    def _assistant(self, state: AgentState) -> dict:
        """Prepare the initial search query."""

        logger.info("RAG node started: assistant")

        return {"search_query": state["question"].strip()}

    def _retrieve(self, state: AgentState) -> dict:
        """Retrieve product documents."""

        started_at = perf_counter()
        logger.info("RAG node started: retrieval")

        try:
            documents = self.retriever.invoke(state["search_query"])

            logger.info(
                "Retrieval returned %d documents",
                len(documents or []),
            )

            context = self._format_documents(documents)

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
        """Generate an answer using retrieved product context."""

        started_at = perf_counter()
        logger.info("RAG node started: generation")

        try:
            context = state["context"]

            if not self._has_context(context):
                logger.info(
                    "No retrieved context; returning fallback answer"
                )
                return {"answer": self.FALLBACK_ANSWER}

            prompt = ChatPromptTemplate.from_template(
                "You are ShopSmart, a helpful e-commerce product assistant.\n"
                "Answer the user's question using only the supplied "
                "product context.\n"
                "Do not invent product names, prices, currency, ratings, "
                "availability, or product details.\n"
                "Respect the user's product requirements and budget.\n"
                "If the context does not contain a suitable product or "
                "requested information, explain this politely.\n"
                "Keep the answer concise and easy to read.\n\n"
                "User question:\n{question}\n\n"
                "Retrieved product context:\n{context}\n\n"
                "Answer:"
            )

            chain = prompt | self.llm | StrOutputParser()

            answer = chain.invoke(
                {
                    "question": state["question"],
                    "context": context,
                }
            ).strip()

            if not answer:
                answer = self.FALLBACK_ANSWER

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
        """Build a minimal workflow with fewer LLM calls."""

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
        """Run the workflow and return the final answer."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        normalized_query = query.strip()

        # Do not invoke the retriever or LLM for simple greetings.
        if self._is_greeting(normalized_query):
            logger.info("Greeting detected; skipping RAG workflow")

            return (
                "Hi! I'm your ShopSmart product assistant. "
                "I can help you find products, compare prices, "
                "and answer questions about our catalog. "
                "What are you looking for?"
            )

        started_at = perf_counter()
        logger.info("Product question received")

        try:
            result = self.app.invoke(
                {
                    "question": normalized_query,
                    "search_query": normalized_query,
                    "context": "",
                    "answer": "",
                },
                config={"recursion_limit": 10},
            )

            answer = result.get("answer", "").strip()

            return answer or self.FALLBACK_ANSWER

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
