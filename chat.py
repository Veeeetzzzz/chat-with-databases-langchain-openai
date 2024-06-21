import os
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}

def get_connection_string(db_type, user, password, host, port, database):
    if db_type.lower() == 'postgresql':
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    elif db_type.lower() == 'mysql':
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError("Unsupported database type. Supported types are 'postgresql' and 'mysql'")

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║                    SQL AI - Chat With Databases                ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    db_type = input("Enter the database type (mysql, postgresql): ")
    user = input("Enter the database username: ")
    password = input("Enter the database password: ")
    host = input("Enter the database host: ")
    port = input("Enter the database port: ")
    database = input("Enter the database name: ")

    try:
        connection_string = get_connection_string(db_type, user, password, host, port, database)
        engine = create_engine(connection_string)
        print("Connected to the database.")
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return

    include_tables = ['employees']

    try:
        db = SQLDatabase(engine, include_tables=include_tables)
        print(f"Included tables: {', '.join(include_tables)}")
    except Exception as e:
        print(f"Failed to include tables: {e}")
        return

    try:
        toolkit = SQLDatabaseToolkit(db=db)
        print("SQLDatabaseToolkit initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize SQLDatabaseToolkit: {e}")
        return

    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_texts = [chunk for table in include_tables for chunk in text_splitter.split_text(table)]
        print("Text splitting completed.")

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents([Document(page_content=text) for text in split_texts], embeddings)
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
        print("Chroma and RetrievalQA initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Chroma or RetrievalQA: {e}")
        return

    print("Entering question loop...")
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        try:
            agent_executor = create_sql_agent(
                llm=OpenAI(temperature=0),
                toolkit=toolkit,
                verbose=True
            )
            print("SQL agent created successfully.")

            response = agent_executor.run(question)
            print(response)
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()