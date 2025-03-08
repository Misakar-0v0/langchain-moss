from langchain_moss import DefaultMossAction
from langchain_moss.example import tools
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

action = DefaultMossAction(tools.__name__)
tools = [action.as_tool()]

llm = ChatOpenAI(model="gpt-4o", streaming=True)
instruction = f"""
# Moss
{action.get_instruction()}

# Instruction

help user as best as you can
"""
agent = create_openai_tools_agent(
    llm,
    tools,
    ChatPromptTemplate.from_messages([
        instruction,
        "Thought:{agent_scratchpad}",
        "{history}",
    ]),
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

with st.container(border=True):
    st.title("Moss Agent Example")
    st.markdown(instruction)

if "history" not in st.session_state:
    st.session_state["history"] = []

history = st.session_state["history"]
st.write(history)
for item in history:
    if item["role"] == "user":
        with st.chat_message("user"):
            st.markdown(item["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(item["content"])

if prompt := st.chat_input():
    history.append({"role": "user", "content": prompt})
    st.session_state["history"] = history
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response = agent_executor.invoke(
            input={
                "history": history,
            },
        )
        st.write(response["output"])
        history.append({"role": "assistant", "content": response['output']})
        st.session_state["history"] = history
