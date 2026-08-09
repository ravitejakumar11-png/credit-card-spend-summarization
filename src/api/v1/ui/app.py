import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NorthStar Credit Card Assistant",
    page_icon="💳",
    layout="wide",
)


st.title("NorthStar Credit Card Assistant")


# ---------------------------------------------------------------------------
# Create conversation thread
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Developer Mode Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.header("Developer Mode")

    developer_mode = st.toggle(
        "Enable Developer Mode",
        value=False,
    )

    if developer_mode:

        st.divider()

        st.subheader("Document Ingestion")

        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx"],
        )

        if st.button(
            "Ingest Document",
            use_container_width=True,
            disabled=uploaded_file is None,
        ):

            try:

                with st.spinner("Uploading and ingesting document..."):

                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/upload/",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        },
                        timeout=600,
                    )

                if response.ok:

                    result = response.json()

                    st.success("Document ingested successfully.")

                    st.json(result)

                else:

                    st.error(
                        f"Ingestion failed: "
                        f"{response.status_code} - "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:

                st.error(f"Could not connect to the FastAPI server: {exc}")

        st.divider()

        st.subheader("Knowledge Base")

        st.warning(
            "Clearing the knowledge base will delete all "
            "ingested documents, chunks and extracted images."
        )

        if st.button(
            "Clear All Ingested Data",
            use_container_width=True,
        ):

            try:

                with st.spinner("Clearing ingested data..."):

                    response = requests.delete(
                        f"{API_BASE_URL}/api/v1/upload/clear",
                        timeout=120,
                    )

                if response.ok:

                    result = response.json()

                    st.success("All ingested data has been cleared.")

                    st.json(result)

                else:

                    st.error(
                        f"Failed to clear data: "
                        f"{response.status_code} - "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:

                st.error(f"Could not connect to the FastAPI server: {exc}")


# ---------------------------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

if query := st.chat_input("Ask your credit card question..."):

    # -----------------------------------------------------------------------
    # Display user message
    # -----------------------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.write(query)

    # -----------------------------------------------------------------------
    # Call FastAPI query endpoint
    # -----------------------------------------------------------------------

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/query/",
                    json={
                        "query": query,
                        "thread_id": st.session_state.thread_id,
                    },
                    timeout=300,
                )

            if response.ok:

                result = response.json()

                answer = result["answer"]

                st.write(answer)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            else:

                error_message = (
                    f"API request failed: "
                    f"{response.status_code} - "
                    f"{response.text}"
                )

                st.error(error_message)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

    except requests.RequestException as exc:

        error_message = f"Could not connect to the FastAPI server: {exc}"

        with st.chat_message("assistant"):

            st.error(error_message)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )
