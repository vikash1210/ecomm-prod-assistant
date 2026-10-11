"""Agentic RAG workflow using LangGraph."""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, START, StateGraph

from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader


class AgentState(TypedDict):
    """State passed between workflow nodes."""

    question: str
    search_query: str
    context: str
    answer: str
    attempts: int
    relevant: bool


class AgenticRAG:
    """Retrieve, assess relevance, rewrite queries, and generate answers."""

    def __init__(
        self,
        retriever=None,
        llm=None,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.retriever = retriever or Retriever()
        self.llm = llm or ModelLoader().load_llm()
        self.max_retries = max_retries
        self.app = self._build_workflow().compile()

    @staticmethod
    def _format_documents(documents) -> str:
        """Convert retrieved documents into readable context."""
        if not documents:
            return "No relevant product documents found."

        chunks = []
        for document in documents:
            metadata = document.metadata or {}
            chunks.append(
                "\n".join(
                    [
                        f"Product: {metadata.get('product_title', 'N/A')}",
                        f"Price: {metadata.get('price', 'N/A')}",
                        f"Rating: {metadata.get('rating', 'N/A')}",
                        f"Details: {document.page_content.strip()}",
                    ]
                )
            )

        return "\n\n---\n\n".join(chunks)

    def _assistant(self, state: AgentState) -> dict:
        return {"search_query": state["question"]}

    def _retrieve(self, state: AgentState) -> dict:
        documents = self.retriever.invoke(state["search_query"])
        return {"context": self._format_documents(documents)}

    def _grade(self, state: AgentState) -> dict:
        prompt = PromptTemplate.from_template(
            "You are a strict relevance grader for an e-commerce assistant.\n"
            "Question: {question}\n"
            "Retrieved context: {context}\n\n"
            "Does the context contain useful information for answering "
            "the question? Reply with only YES or NO."
        )
        result = (prompt | self.llm | StrOutputParser()).invoke(
            {
                "question": state["question"],
                "context": state["context"],
            }
        )
        return {"relevant": result.strip().upper().startswith("YES")}

    def _rewrite(self, state: AgentState) -> dict:
        prompt = ChatPromptTemplate.from_template(
            "Rewrite the search query for an e-commerce product catalog.\n"
            "Preserve the user's intent, product names, budget, and constraints.\n"
            "Do not answer the question. Return only the rewritten query.\n\n"
            "Original question: {question}\n"
            "Previous search query: {search_query}"
        )
        rewritten = (
            prompt | self.llm | StrOutputParser()
        ).invoke(
            {
                "question": state["question"],
                "search_query": state["search_query"],
            }
        ).strip()

        return {
            "search_query": rewritten or state["question"],
            "attempts": state["attempts"] + 1,
        }

    def _generate(self, state: AgentState) -> dict:
        prompt = ChatPromptTemplate.from_template(
            "You are an e-commerce product assistant.\n"
            "Answer using only the supplied context. Do not invent prices, "
            "ratings, availability, or product details.\n"
            "If the context does not contain the answer, say so clearly.\n\n"
            "Question: {question}\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )
        answer = (
            prompt | self.llm | StrOutputParser()
        ).invoke(
            {
                "question": state["question"],
                "context": state["context"],
            }
        )
        return {"answer": answer}

    def _route_after_grade(self, state: AgentState) -> str:
        if state["relevant"] or state["attempts"] >= self.max_retries:
            return "generate"
        return "rewrite"

    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("assistant", self._assistant)
        workflow.add_node("retriever", self._retrieve)
        workflow.add_node("grader", self._grade)
        workflow.add_node("rewriter", self._rewrite)
        workflow.add_node("generator", self._generate)

        workflow.add_edge(START, "assistant")
        workflow.add_edge("assistant", "retriever")
        workflow.add_edge("retriever", "grader")
        workflow.add_conditional_edges(
            "grader",
            self._route_after_grade,
            {"generate": "generator", "rewrite": "rewriter"},
        )
        workflow.add_edge("rewriter", "retriever")
        workflow.add_edge("generator", END)
        return workflow

    def run(self, query: str) -> str:
        """Run the workflow for a non-empty user query."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        result = self.app.invoke(
            {
                "question": query.strip(),
                "search_query": query.strip(),
                "context": "",
                "answer": "",
                "attempts": 0,
                "relevant": False,
            }
        )
        return result["answer"]


if __name__ == "__main__":
    agent = AgenticRAG()
    print(agent.run("What is the price of iPhone 15?"))