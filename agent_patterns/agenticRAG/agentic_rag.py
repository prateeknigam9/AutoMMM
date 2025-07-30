from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from rich import print
from pydantic import BaseModel, Field
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from utils import utility
from utils.memory_handler import DataStore
import json
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama

AgenticRagPrompts = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "AgenticRagPrompts",
)


class AgenticRag:
    def __init__(self, memory_folder_path : str) -> None:
        self.llm = ChatOllama(model = "llama3.1")
        self.embedding_model="nomic-embed-text"
        self.retriever_tool = self.load_as_tool()
        self.graph = self._build_graph_(state=MessagesState)
    
    def get_text_chunks_langchain(self,text):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
        return docs

    def save_to_vectorStore(self, model:str):
        memory_context = DataStore.get_str("memory_context")        
        docs = self.get_text_chunks_langchain(memory_context)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=OllamaEmbeddings(model=model),
            persist_directory="rag_memory_db",
        )
        return vectorstore
    
    def load_as_tool(self):
        vectorstore = self.save_to_vectorStore(model=self.embedding_model)
        retriever = vectorstore.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            "retrive_context",
            "Search and return information from the memory context",
        )
        return retriever_tool


    def generate_query_or_respond(self, state: MessagesState):
        response = (
            self.llm
            .bind_tools([self.retriever_tool]).invoke(state["messages"])
        )
        return {"messages": [response]}


    def grade_documents(self,
        state: MessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = AgenticRagPrompts['GRADE_PROMPT'].format(question=question, context=context)

        class GradeDocuments(BaseModel):
            binary_score: str = Field(
                description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
            )

        response = (
            self.llm
            .with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"


    def rewrite_question(self,state: MessagesState):
        messages = state["messages"]
        question = messages[0].content
        prompt = AgenticRagPrompts['REWRITE_PROMPT'].format(question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}


    def generate_answer(self, state: MessagesState):
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = AgenticRagPrompts['GENERATE_PROMPT'].format(question=question, context=context)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}


    def _build_graph(self, state: MessagesState):
        workflow = StateGraph(MessagesState)

        # Define the nodes we will cycle between
        workflow.add_node(self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node(self.rewrite_question)
        workflow.add_node(self.generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile
        return workflow.compile()

    def relevant_doc_extract(self, state:MessagesState):
        vectorstore = self.save_to_vectorStore(model=self.embedding_model)
        retriever = vectorstore.as_retriever()
        results = retriever.get_relevant_documents(state['messages'][0].content)
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(doc.page_content)

    def _build_graph_(self, state: MessagesState):
        workflow = StateGraph(MessagesState)
        workflow.add_node("relevant_doc_extract",self.relevant_doc_extract)

        workflow.add_edge(START, "relevant_doc_extract")
        workflow.add_edge("relevant_doc_extract",END)
        return workflow.compile()

#  Convert to Natural language first