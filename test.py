import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from langchain_core.runnables.config import RunnableConfig
from azure.identity import DefaultAzureCredential
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Dict
import torch
import json
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from sqlalchemy import text, inspect

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///example.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ChromaDB Configuration
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "schema_embeddings"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# SQLAlchemy Base Model
Base = declarative_base()

# -------------------------- Database Table Definitions -------------------------- #

class User(Base):
    """Represents a user in the system."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    email = Column(String, unique=True, index=True)
    orders = relationship("Order", back_populates="user")


# -------------------------- Agent State Definition -------------------------- #

class AgentState(BaseModel):
    question: str
    sql_query: Optional[str] = None
    query_result: Optional[str] = None
    query_rows: List[dict] = []
    current_user: str = ""
    attempts: int = 0
    relevance: str = ""
    sql_error: bool = False
    context: List[str] = []
    chart_type: Optional[str] = None
    chart_title: Optional[str] = None

MAX_RETRIES = 3


# -------------------------- Schema Extraction and Processing -------------------------- #
from typing import Dict, List
import torch



class ThaiBGEEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function with proper pooling and Thai language support
    """
    def __init__(self, model_name= "BAAI/bge-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Disable dropout for evaluation

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input texts
        Args:
            texts (List[str]): List of strings to embed
        Returns:
            List[List[float]]: List of embeddings
        """
        if isinstance(texts, str):  # Ensure input is a list
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS pooling
        return embeddings.cpu().numpy().tolist()

def process_schema_to_graph_and_embeddings(schema: Dict, chroma_collection, embedding_model) -> nx.DiGraph:
    """
    Process database schema into a graph structure and store embeddings in ChromaDB,
    using M-Schema format for enhanced metadata.
    """
    graph = nx.DiGraph()
    entries = []

    for table_name, details in schema.items():
        # Add table as a node with enriched metadata
        graph.add_node(
            table_name,
            type="table",
            description=details.get("description", ""),
            primary_keys=details.get("primary_keys", []),
            examples=details.get("examples", {})
        )

        # Process columns with M-Schema metadata
        for column in details["columns"]:
            column_id = f"{table_name}.{column['name']}"
            
            # Safely get examples for the column and serialize them
            examples = column.get("examples", [])
            examples_serialized = ", ".join(map(str, examples)) if examples else "None"
            
            graph.add_node(
                column_id,
                type="column",
                table=table_name,
                data_type=column["type"],
                primary_key=column.get("primary_key", False),
                examples=examples_serialized  # Store serialized examples
            )
            graph.add_edge(table_name, column_id, relation="has_column")

            # Add column embedding entry
            column_text = (
                f"Column: {column['name']} ({column['type']}), "
                f"Primary Key: {column.get('primary_key', False)}, "
                f"Examples: {examples_serialized}"
            )
            embedding = embedding_model([column_text])[0]

            entries.append({
                "id": column_id,
                "text": column_text,
                "embedding": embedding,
                "metadata": {
                    "type": "column",
                    "table": table_name,
                    "data_type": column["type"],
                    "primary_key": column.get("primary_key", False),
                    "examples": examples_serialized  # Store serialized examples in metadata
                }
            })

        # Process relationships with enriched metadata
        for relation in details.get("relationships", []):
            rel_id = f"{table_name}.{relation['source_column']}->{relation['target_table']}.{relation['target_column']}"
            
            graph.add_edge(
                table_name,
                relation["target_table"],
                relation="foreign_key",
                source_column=relation["source_column"],
                target_column=relation["target_column"]
            )
            
            # Relationship text for embeddings
            rel_text = (
                f"Foreign Key: {table_name}.{relation['source_column']} → "
                f"{relation['target_table']}.{relation['target_column']}, "
                f"Maps to: {relation.get('maps_to', '')}"
            )
            embedding = embedding_model([rel_text])[0]

            entries.append({
                "id": rel_id,
                "text": rel_text,
                "embedding": embedding,
                "metadata": {
                    "type": "relationship",
                    "source_table": table_name,
                    "source_column": relation["source_column"],
                    "target_table": relation["target_table"],
                    "target_column": relation["target_column"],
                    "maps_to": relation.get("maps_to", "")
                }
            })

    # Debug: Print sample metadata for verification
    if entries:
        print(f"Sample Metadata Entry: {entries[0]['metadata']}")

    # Store embeddings in ChromaDB
    if entries:
        chroma_collection.add(
            ids=[entry["id"] for entry in entries],
            documents=[entry["text"] for entry in entries],
            embeddings=[entry["embedding"] for entry in entries],
            metadatas=[entry["metadata"] for entry in entries]
        )
    
    return graph


# -------------------------- ENHANCED RETRIEVAL ------------------------- #
def retrieve_context(state: AgentState):
    state["context"] = hybrid_search(
        state["question"],
        schema,
        chroma_collection,
        embedding_model,
        graph,
        max_results=15
    )
    return state


from sqlalchemy.sql import text  # Import text for raw SQL queries

def extract_database_schema(engine):
    inspector = inspect(engine)
    schema = {}

    # Create a connection for executing SQL queries
    with engine.connect() as connection:
        tables = inspector.get_table_names()

        for table_name in tables:
            schema[table_name] = {"columns": [], "relationships": [], "examples": {}}
            
            # Extract columns
            columns = inspector.get_columns(table_name)
            for column in columns:
                # Add column details with examples
                examples = []  # Retrieve examples from the database
                try:
                    # Use sqlalchemy.text to create a valid SQL query object
                    query = text(f"SELECT {column['name']} FROM {table_name} LIMIT 3")
                    examples = [row[0] for row in connection.execute(query).fetchall()]
                except Exception as e:
                    print(f"Failed to fetch examples for {table_name}.{column['name']}: {e}")
                
                schema[table_name]["columns"].append({
                    "name": column["name"],
                    "type": str(column["type"]).split("(")[0],
                    "examples": examples
                })

            # Extract foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                schema[table_name]["relationships"].append({
                    "source_column": fk["constrained_columns"][0],
                    "target_table": fk["referred_table"],
                    "target_column": fk["referred_columns"][0]
                })

    return schema

def keyword_search(query: str, schema: Dict) -> List[str]:
    """
    Perform keyword search to find schema elements directly matching query keywords.
    Args:
        query (str): User's natural language query.
        schema (Dict): Extracted database schema.
    Returns:
        List[str]: List of schema elements matching the query keywords.
    """
    keywords = query.lower().split()  # Split the query into individual words
    matches = []

    for table, details in schema.items():
        # Match table names
        if any(keyword in table.lower() for keyword in keywords):
            matches.append(table)

        # Match column names
        for column in details["columns"]:
            if any(keyword in column["name"].lower() for keyword in keywords):
                matches.append(f"{table}.{column['name']}")

    return matches



def semantic_search(query: str, chroma_collection, embedding_model, max_results: int = 20) -> List[str]:
    """
    Perform semantic search to retrieve schema elements based on embeddings.
    Args:
        query (str): User's natural language query.
        chroma_collection: ChromaDB collection.
        embedding_model: Embedding model for generating embeddings.
        max_results (int): Maximum number of results to return.
    Returns:
        List[str]: List of schema elements matching the query semantically.
    """
    # Generate embedding for the query
    query_embedding = embedding_model([query])[0]
    
    # Query ChromaDB
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=max_results)
    
    # Extract IDs of matching schema elements
    return results["ids"][0]


def hybrid_search(
    query: str,
    schema: Dict,
    chroma_collection,
    embedding_model,
    graph: nx.DiGraph,
    max_results: int = 20
) -> List[str]:
    """
    Combine keyword search and semantic search for hybrid retrieval.
    Args:
        query (str): User's natural language query.
        schema (Dict): Extracted database schema.
        chroma_collection: ChromaDB collection.
        embedding_model: Embedding model for generating embeddings.
        max_results (int): Maximum number of results to return.
    Returns:
        List[str]: Combined results from keyword and semantic search.
    """
    # Perform keyword search
    keyword_matches = keyword_search(query, schema)

    # Perform semantic search
    semantic_matches = semantic_search(query, chroma_collection, embedding_model, max_results)

    # Combine and deduplicate results
    combined_matches = list(set(keyword_matches + semantic_matches))

    # Limit results to max_results
    return combined_matches[:max_results]


def assemble_prompt(query: str, context: List[str], graph: nx.DiGraph) -> str:
    """
    Build a structured and optimized prompt with schema context, highlighting relevant parts
    and including enriched metadata like examples and primary keys.
    """
    prompt_lines = [
        "### Task",
        "Translate the following natural language query into an SQL query:",
        "",
        f"### Query\n{query}",
        "",
        "### Relevant Schema Information"
    ]

    # Collect relevant tables and their metadata
    relevant_tables = set()
    for node in context:
        try:
            if graph.nodes[node]["type"] == "table":
                relevant_tables.add(node)
            elif graph.nodes[node]["type"] == "column":
                relevant_tables.add(graph.nodes[node]["table"])
        except KeyError:
            continue  # Skip nodes that lack necessary attributes

    # Add details for relevant tables and columns
    for table in relevant_tables:
        table_metadata = graph.nodes[table]
        primary_keys = table_metadata.get("primary_keys", [])
        examples = table_metadata.get("examples", {})
        prompt_lines.append(f"\nTable: {table}")
        if primary_keys:
            prompt_lines.append(f"  - Primary Keys: {', '.join(primary_keys)}")
        if examples:
            prompt_lines.append(f"  - Examples: {examples}")

        # Add column details
        for neighbor in graph.neighbors(table):
            try:
                if graph.nodes[neighbor]["type"] == "column":
                    column_metadata = graph.nodes[neighbor]
                    column_name = neighbor.split(".")[1]
                    column_type = column_metadata["data_type"]
                    primary_key = column_metadata.get("primary_key", False)
                    column_examples = column_metadata.get("examples", [])

                    # Highlight relevant columns
                    if neighbor in context:
                        prompt_lines.append(
                            f"  - {column_name} ({column_type}) [RELEVANT] - "
                            f"{'Primary Key' if primary_key else ''} "
                            f"{f'Examples: {column_examples}' if column_examples else ''}"
                        )
                    else:
                        prompt_lines.append(
                            f"  - {column_name} ({column_type}) - "
                            f"{'Primary Key' if primary_key else ''} "
                            f"{f'Examples: {column_examples}' if column_examples else ''}"
                        )
            except KeyError:
                continue

    # Add relevant relationships
    prompt_lines.append("\n### Relationships")
    for edge in graph.edges(data=True):
        if edge[2].get("relation") == "foreign_key":
            source = edge[0]
            target = edge[1]
            source_col = edge[2]["source_column"]
            target_col = edge[2]["target_column"]
            maps_to = edge[2].get("maps_to", "")
            relationship_examples = edge[2].get("examples", [])
            if source in relevant_tables or target in relevant_tables:
                prompt_lines.append(
                    f"  - {source} ({source_col}) → {target} ({target_col}) [RELEVANT] "
                    f"{f'Maps to: {maps_to}' if maps_to else ''} "
                    f"{f'Examples: {relationship_examples}' if relationship_examples else ''}"
                )

    # Final instructions
    prompt_lines.append("\n### SQL Query")
    prompt_lines.append("-- Write the SQL query below this line --")
    print(f"Assemble Prompt: {prompt_lines}")
    return "\n".join(prompt_lines)


COLLECTION_NAME = 'schema_embeddings'

def setup_chroma_db():
    """
    Initialize ChromaDB and ensure collection matches embedding dimension
    """
    schema = extract_database_schema(engine)
    
    chroma_client = PersistentClient(path=CHROMA_DB_DIR)
    try:
        # Try to get the collection
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        # Create collection with correct embedding function and dimension
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ThaiBGEEmbeddingFunction()
        )
    return chroma_client, collection



schema = extract_database_schema(engine)
chroma_client, chroma_collection = setup_chroma_db()
embedding_model = ThaiBGEEmbeddingFunction()
graph = process_schema_to_graph_and_embeddings(schema, chroma_collection, embedding_model)

from langchain_core.prompts.chat import ChatPromptTemplate

def check_relevance(state: AgentState, config: dict):
    """Determine if the user's question is relevant to the database schema using context-based retrieval."""
    question = state["question"]
    
    # Retrieve relevant schema context using hybrid search
    state["context"] = hybrid_search(
        question,
        schema,  # Extracted database schema
        chroma_collection,
        embedding_model,
        graph,
        max_results=15  # Retrieve top 15 relevant elements
    )

    # If no relevant schema elements were found, mark as 'not_relevant'
    if not state["context"]:
        state["relevance"] = "not_relevant"
        print("[DEBUG] No relevant schema context found. Marking as not relevant.")
        return state
    print(f"[DEBUG] Retrieved relevant schema context: {state['context']}")
    # Construct prompt with retrieved schema context
    system_prompt = f"""
    You are an intelligent assistant determining if a given question is related to a database schema.

    - If the question is **clearly related** to the database (queries about data, tables, or relationships), respond `"relevant"`.
    - If the question is **unrelated** (about the weather, general facts, jokes, etc.), respond `"not_relevant"`.

    **Database Schema Context (Most Relevant Retrieved Entries):**
    {state["context"]}

    **Example Scenarios:**
    - User Question: "What is the weather today?"
      Response: "not_relevant"
    - User Question: "Show me all coupons and their discount."
      Response: "relevant"

    **User Question:**
    {question}
    """

    # LLM call to determine relevance
    llm = AzureChatOpenAI(
        temperature=0,
        azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME")
    )

    response = llm([SystemMessage(content=system_prompt)])

    # Extract response and update state
    state["relevance"] = response.content.strip().lower()
    print(f"[DEBUG] LLM determined relevance as: {state['relevance']}")

    return state


def setup_chroma_db():
    """
    Initialize ChromaDB and ensure collection matches embedding dimension
    """
    schema = extract_database_schema(engine)
    
    chroma_client = PersistentClient(path=CHROMA_DB_DIR)
    try:
        # Try to get the collection
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        # Create collection with correct embedding function and dimension
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ThaiBGEEmbeddingFunction()
        )
    return chroma_client, collection

def extract_database_schema(engine):
    """Extracts database schema details for use in SQL generation."""
    inspector = inspect(engine)
    schema = {}
    with engine.connect() as connection:
        tables = inspector.get_table_names()
        for table_name in tables:
            schema[table_name] = {"columns": [], "relationships": [], "examples": {}}
            columns = inspector.get_columns(table_name)
            for column in columns:
                examples = []
                try:
                    query = text(f"SELECT {column['name']} FROM {table_name} LIMIT 3")
                    examples = [row[0] for row in connection.execute(query).fetchall()]
                except Exception as e:
                    print(f"Failed to fetch examples for {table_name}.{column['name']}: {e}")
                schema[table_name]["columns"].append({
                    "name": column["name"],
                    "type": str(column["type"]).split("(")[0],
                    "examples": examples
                })
            foreign_keys = inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                schema[table_name]["relationships"].append({
                    "source_column": fk["constrained_columns"][0],
                    "target_table": fk["referred_table"],
                    "target_column": fk["referred_columns"][0]
                })
    return schema


# -------------------------- Schema Context Retrieval -------------------------- #


def get_database_schema(context, graph):
    """
    Extracts relevant schema information based on the provided context.
    
    Args:
        context (List[str]): List of relevant schema elements retrieved via hybrid search.
        graph (nx.DiGraph): Graph representation of the database schema.
    
    Returns:
        str: A formatted string containing relevant schema information.
    """
    prompt_lines = ["### Relevant Schema Information"]
    
    relevant_tables = set()
    
    # Identify relevant tables from context
    for node in context:
        try:
            if graph.nodes[node]["type"] == "table":
                relevant_tables.add(node)
            elif graph.nodes[node]["type"] == "column":
                relevant_tables.add(graph.nodes[node]["table"])
        except KeyError:
            continue  # Skip nodes that lack necessary attributes
    
    # Add table details
    for table in relevant_tables:
        table_metadata = graph.nodes[table]
        primary_keys = table_metadata.get("primary_keys", [])
        examples = table_metadata.get("examples", {})
        prompt_lines.append(f"\nTable: {table}")
        if primary_keys:
            prompt_lines.append(f"  - Primary Keys: {', '.join(primary_keys)}")
        if examples:
            prompt_lines.append(f"  - Examples: {examples}")

        # Add column details
        for neighbor in graph.neighbors(table):
            try:
                if graph.nodes[neighbor]["type"] == "column":
                    column_metadata = graph.nodes[neighbor]
                    column_name = neighbor.split(".")[1]
                    column_type = column_metadata["data_type"]
                    primary_key = column_metadata.get("primary_key", False)
                    column_examples = column_metadata.get("examples", [])

                    # Highlight relevant columns
                    if neighbor in context:
                        prompt_lines.append(
                            f"  - {column_name} ({column_type}) [RELEVANT] - "
                            f"{'Primary Key' if primary_key else ''} "
                            f"{f'Examples: {column_examples}' if column_examples else ''}"
                        )
                    else:
                        prompt_lines.append(
                            f"  - {column_name} ({column_type}) - "
                            f"{'Primary Key' if primary_key else ''} "
                            f"{f'Examples: {column_examples}' if column_examples else ''}"
                        )
            except KeyError:
                continue

    # Add relevant relationships
    prompt_lines.append("\n### Relationships")
    for edge in graph.edges(data=True):
        if edge[2].get("relation") == "foreign_key":
            source = edge[0]
            target = edge[1]
            source_col = edge[2]["source_column"]
            target_col = edge[2]["target_column"]
            maps_to = edge[2].get("maps_to", "")
            relationship_examples = edge[2].get("examples", [])
            if source in relevant_tables or target in relevant_tables:
                prompt_lines.append(
                    f"  - {source} ({source_col}) → {target} ({target_col}) [RELEVANT] "
                    f"{f'Maps to: {maps_to}' if maps_to else ''} "
                    f"{f'Examples: {relationship_examples}' if relationship_examples else ''}"
                )

    return "\n".join(prompt_lines)


