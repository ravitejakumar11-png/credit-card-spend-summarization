import streamlit as st
import uuid

from langchain_core.messages import HumanMessage

from src.api.v1.agents.agents import rag_graph

st.title("NorthStar Credit Card Assistant")


# ------------------------------------
# Create conversation thread
# ------------------------------------

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())


# ------------------------------------
# Display previous chat messages
# ------------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ------------------------------------
# User input
# ------------------------------------

if query := st.chat_input("Ask your credit card question..."):

    # display user message

    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):

        st.write(query)

    # --------------------------------
    # LangGraph invocation
    # --------------------------------

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    response = rag_graph.invoke(
        {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "evaluate_count": 0,
        },
        config=config,
    )

    answer = response["response"]["answer"]

    # --------------------------------
    # Display assistant response
    # --------------------------------

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):

        st.write(answer)
