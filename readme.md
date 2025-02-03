text-to-sql-agent/
│── src/
│   ├── __init__.py
│   ├── main.py                      # Entry point for running the agent
│   ├── config.py                     # Configuration settings and environment variables
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py                 # SQLAlchemy ORM models
│   │   ├── db_connection.py          # Database connection setup
│   │   ├── schema_extractor.py       # Extract database schema for LLM
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedding_model.py        # ThaiBGE embedding function
│   │   ├── chroma_manager.py         # Handles ChromaDB operations
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py          # Keyword + semantic search for schema retrieval
│   │   ├── schema_graph.py           # Schema graph processing with NetworkX
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state_management.py       # Defines `AgentState`
│   │   ├── query_generation.py       # Convert NL to SQL
│   │   ├── sql_execution.py          # SQL execution and error handling
│   │   ├── query_refinement.py       # Query improvement & debugging
│   │   ├── workflow.py               # Defines the agent's workflow using `StateGraph`
│   │   ├── visualization.py          # Handles chart generation logic
│   │   ├── response_generator.py     # Converts query results into human-readable output
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_utils.py          # Logging & debugging functions
│   │   ├── prompt_templates.py       # Stores LLM prompt templates
│   
│── data/
│   ├── chroma_db/                    # Persistent ChromaDB storage
│── notebooks/
│   ├── development.ipynb              # Jupyter notebook for testing
│── .env                               # Environment variables
│── requirements.txt                    # Python dependencies
│── README.md                           # Documentation
│── setup.py                            # Installation script (if needed)