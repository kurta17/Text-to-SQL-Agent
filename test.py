import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy import text, inspect
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///example.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define database models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    email = Column(String, unique=True, index=True)

class Food(Base):
    __tablename__ = "food"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    price = Column(Float)

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    food_id = Column(Integer, ForeignKey("food.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

# Define agent state
class AgentState(BaseModel):
    current_user: str = ""
    question: str = ""
    relevance: str = ""
    sql_query: str = ""
    query_rows: list = []
    query_result: str = ""
    
class CheckRelevance(BaseModel):
    relevance: str = Field(description="Is the question relevant to the database?")
# Utility function to get database schema
def get_database_schema(engine):
    inspector = inspect(engine)
    schema = ""
    for table_name in inspector.get_table_names():
        schema += f"Table: {table_name}\n"
        for column in inspector.get_columns(table_name):
            col_name = column["name"]
            col_type = str(column["type"])
            if column.get("primary_key"):
                col_type += ", Primary Key"
            if column.get("foreign_keys"):
                fk = list(column["foreign_keys"])[0]
                col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
            schema += f"- {col_name}: {col_type}\n"
        schema += "\n"
    print(f"[DEBUG] Database Schema:\n{schema}")
    return schema

# Workflow steps
def get_current_user(state: AgentState, config: RunnableConfig):
    user_id = config.get("configurable", {}).get("current_user_id", None)
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == int(user_id)).first()
        state.current_user = user.name if user else "User not found"
        print(f"[DEBUG] Current user set to: {state.current_user}")
    finally:
        session.close()
    return state

def check_relevance(state: AgentState, config: RunnableConfig):
    question = state.question
    schema = get_database_schema(engine)
    print(f"[DEBUG] Checking relevance of the question: {question}")
    system = f"""
    You are an assistant that determines whether a given question is related to the following database schema.

    Schema:
    {schema}

    Respond with only \"relevant\" or \"not_relevant\".
    """
    human = f"Question: {question}"
    check_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(CheckRelevance)
    try:
        relevance_checker = check_prompt | structured_llm
        relevance = relevance_checker.invoke({})
        state.relevance = relevance.relevance
        print(f"[DEBUG] Relevance determined: {state.relevance}")
    except Exception as e:
        print(f"[ERROR] Error during relevance check: {e}")
        state.relevance = "not_relevant"
    return state

def convert_nl_to_sql(state: AgentState, config: RunnableConfig):
    if state.relevance != "relevant":
        print("[DEBUG] Question marked as not relevant")
        state.sql_query = "The question is not relevant to the database."
        return state

    schema = get_database_schema(engine)
    question = state.question
    current_user = state.current_user
    system = f"""
    You are an assistant that converts natural language questions into SQL queries based on the following schema:

    Schema:
    {schema}

    The current user is '{current_user}'. Ensure that all query-related data is scoped to this user.

    Provide only the SQL query without any explanations.
    """
    human = f"Question: {question}"
    convert_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    llm = ChatOpenAI(temperature=0)
    sql_generator = convert_prompt | StrOutputParser()

    try:
        response = sql_generator.invoke({})
        if isinstance(response, str):
            state.sql_query = response.strip()
        else:
            raise ValueError("Unexpected response format from LLM.")
        print(f"[DEBUG] SQL Query Generated: {state.sql_query}")
    except Exception as e:
        print(f"[ERROR] Error in convert_nl_to_sql: {e}")
        state.sql_query = "Error generating SQL query."
    return state

def execute_sql(state: AgentState):
    if "Error" in state.sql_query or "not relevant" in state.sql_query:
        print("[DEBUG] Skipping execution: Invalid SQL query.")
        state.query_rows = []
        return state

    session = SessionLocal()
    try:
        print(f"[DEBUG] Executing SQL Query: {state.sql_query}")
        result = session.execute(text(state.sql_query))
        state.query_rows = result.fetchall()
        print(f"[DEBUG] Query Rows: {state.query_rows}")
    except Exception as e:
        print(f"[ERROR] Error executing SQL query: {e}")
        state.query_rows = []
    finally:
        session.close()
    return state

def generate_human_readable_answer(state: AgentState):
    rows = state.query_rows
    current_user = state.current_user
    state.query_result = (
        f"Hello {current_user}, found {len(rows)} rows." if rows else f"Hello {current_user}, no data found."
    )
    print(f"[DEBUG] Generated Answer: {state.query_result}")
    return state

# Workflow definition
class TextToSQLFlow:
    def __init__(self):
        self.state = AgentState()
        self.config = {}

    def kickoff(self):
        self.state = get_current_user(self.state, self.config)
        self.state = check_relevance(self.state, self.config)
        self.state = convert_nl_to_sql(self.state, self.config)
        self.state = execute_sql(self.state)
        self.state = generate_human_readable_answer(self.state)
        return self.state

# Example usage
if __name__ == "__main__":
    flow = TextToSQLFlow()
    flow.state.question = "Show me all delivery information for user 10 and orders."
    flow.config = {"configurable": {"current_user_id": "10"}}
    final_state = flow.kickoff()
    print("Final Output:", final_state.query_result)
