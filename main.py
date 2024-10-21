from flask import Flask, request, jsonify
from langchain_community.llms import OpenAI 
from langchain_community.utilities import SQLDatabase
#from langchain.sql_database import SQLDatabase
#from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
#from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
import os
from urllib.parse import quote_plus
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain.agents.agent_types import AgentType

app = Flask(__name__)

# Set up database connection
db_username = os.getenv("DB_USERNAME", "postgres")
db_password = os.getenv("DB_PASSWORD", "your_password")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "database_name")

# Properly encode the password
encoded_password = quote_plus(db_password)

# Construct the connection string
db_url = f"postgresql://{db_username}:{encoded_password}@{db_host}:{db_port}/{db_name}"


#db_url = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
db = SQLDatabase.from_uri(db_url)

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = 'api_key'

# Create SQL agent
llm = ChatOpenAI(model="gpt-4o-mini")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
system_message = prompt_template.format(dialect="SQLite", top_k=5)


agent_executor = create_react_agent(
    llm, toolkit.get_tools(), state_modifier=system_message
)

# agent_executor = create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=False,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     state_modifier=system_message
# )


#agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)




# Example route to handle SQL agent requests
@app.route('/query', methods=['POST'])
def query_database():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        #return agent_executor.run(query)
        events = agent_executor.stream(
            {"messages": [("user", query)]},
            stream_mode="values",
        )

        # Collect the results from the streaming response
        response_data = []
        for event in events:
            last_message = event["messages"][-1]
            # Check if the message has a 'content' or 'text' attribute
            if hasattr(last_message, 'content'):
                response_data.append(last_message.content)
            elif hasattr(last_message, 'text'):
                response_data.append(last_message.text)
            else:
                response_data.append(str(last_message))

        # Return the result as JSON
        return jsonify({"result": response_data[-1]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
