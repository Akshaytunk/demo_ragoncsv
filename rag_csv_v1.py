from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
import os
from langchain.prompts import PromptTemplate

st.title("RAG on csv file")
#file = "C:/Users/Family/Downloads/dataset_dummy.csv"
with st.sidebar :
    file = st.file_uploader("upload csv file")

df = pd.read_csv(file)
#df["Date"]=pd.to_datetime(df["Date"])
query = st.text_input("enter your query")
template = '''You are a helpful data analyst assistant.
 
You are working with a Pandas DataFrame named `df` that has the following columns:
- `Account`:describes the type of metric (e.g., 'VOLUME', 'Options MSRP', 'Gross Margin')
- `Date`: an integer in YYYYMM format (e.g., 202401 for January 2024)
- `Value`: numeric value
 
The data is in long format, where multiple metrics are stored as rows under the `Bonus_AccountsId` column.
 
 
### Instructions:
1. When the user asks about a metric group (e.g., "VOLUME"), filter the `Account` column to only include those related metrics.
2. Use pandas to group the data by the appropriate dimension (e.g., `Date`) as implied in the user request.
3. Use the appropriate aggregate function based on the user query. This may include:
   - `sum` (e.g., total volume)
   - `average` (e.g., average spend per market)
   - `min`, `max`, or `count` (e.g., number of entries)
4. Return the result as a natural language answer, summarizing the key findings in full sentences (do not show raw code or DataFrame unless explicitly asked).
5. Include values, comparisons, and dates if possible to make the summary insightful.Do not return code, return result of code in natural language
 
Now, analyze and respond to this user query using pandas and natural language:
 
**User question:** {user_query}
'''

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4",api_key=os.getenv("OPENAI_API_KEY")),
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
)

prompt = PromptTemplate(template=template)
formate_prompt = prompt.format (user_query = query)
response = agent.invoke(formate_prompt)
print(response)
st.write(response["output"])