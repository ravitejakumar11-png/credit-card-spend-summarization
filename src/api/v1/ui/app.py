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
# Delete / Clear / Ingestion state
# ---------------------------------------------------------------------------

if "confirm_delete_doc_id" not in st.session_state:
    st.session_state.confirm_delete_doc_id = None

if "confirm_delete_filename" not in st.session_state:
    st.session_state.confirm_delete_filename = None

if "delete_message" not in st.session_state:
    st.session_state.delete_message = None

if "confirm_clear_all" not in st.session_state:
    st.session_state.confirm_clear_all = False

if "clear_message" not in st.session_state:
    st.session_state.clear_message = None

# ---------------------------------------------------------------------------
# Ingestion result state
# ---------------------------------------------------------------------------

if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = None

if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None

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

        # ================================================================
        # Document Ingestion
        # ================================================================

        st.subheader("Document Ingestion")

        # ---------------------------------------------------------------
        # Display previous ingestion result
        # ---------------------------------------------------------------

        if st.session_state.ingest_message:

            st.success(
                st.session_state.ingest_message
            )

            if st.session_state.ingest_result:

                st.json(
                    st.session_state.ingest_result
                )

            # Clear the persisted result after displaying it.
            st.session_state.ingest_message = None
            st.session_state.ingest_result = None

        # ---------------------------------------------------------------
        # File upload
        # ---------------------------------------------------------------

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

                with st.spinner(
                    "Uploading and ingesting document..."
                ):

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

                    # ---------------------------------------------------
                    # Persist ingestion response before rerun.
                    # ---------------------------------------------------

                    st.session_state.ingest_message = (
                        "Document ingested successfully."
                    )

                    st.session_state.ingest_result = result

                    st.rerun()

                else:

                    st.error(
                        f"Ingestion failed: "
                        f"{response.status_code} - "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:

                st.error(
                    f"Could not connect to the FastAPI server: {exc}"
                )

        st.divider()

        # ================================================================
        # Knowledge Base
        # ================================================================

        st.subheader("Knowledge Base")

        # ---------------------------------------------------------------
        # Display individual document deletion message
        # ---------------------------------------------------------------

        if st.session_state.delete_message:

            st.success(
                st.session_state.delete_message
            )

            st.session_state.delete_message = None

        # ---------------------------------------------------------------
        # Display clear-all success message
        # ---------------------------------------------------------------

        if st.session_state.clear_message:

            st.success(
                st.session_state.clear_message
            )

            st.session_state.clear_message = None

        # ---------------------------------------------------------------
        # Get currently ingested documents
        # ---------------------------------------------------------------

        documents = []

        try:

            with st.spinner("Loading documents..."):

                response = requests.get(
                    f"{API_BASE_URL}/api/v1/upload/",
                    timeout=30,
                )

            if response.ok:

                result = response.json()

                documents = result.get(
                    "documents",
                    [],
                )

            else:

                st.error(
                    f"Failed to load documents: "
                    f"{response.status_code} - "
                    f"{response.text}"
                )

        except requests.RequestException as exc:

            st.error(
                f"Could not connect to the FastAPI server: {exc}"
            )

        # ---------------------------------------------------------------
        # Delete individual document
        # ---------------------------------------------------------------

        st.markdown("**Delete Document**")

        if documents:

            document_options = {
                document["filename"]: document
                for document in documents
            }

            selected_filename = st.selectbox(
                "Select document",
                options=list(document_options.keys()),
            )

            selected_document = document_options[
                selected_filename
            ]

            selected_doc_id = str(
                selected_document["id"]
            )

            st.caption(
                f"Document ID: {selected_doc_id}"
            )

            # -----------------------------------------------------------
            # Check whether this document is awaiting confirmation
            # -----------------------------------------------------------

            confirmation_active = (
                st.session_state.confirm_delete_doc_id
                == selected_doc_id
            )

            # -----------------------------------------------------------
            # Normal delete button
            # -----------------------------------------------------------

            if not confirmation_active:

                if st.button(
                    "Delete Selected Document",
                    use_container_width=True,
                ):

                    st.session_state.confirm_delete_doc_id = (
                        selected_doc_id
                    )

                    st.session_state.confirm_delete_filename = (
                        selected_filename
                    )

                    st.rerun()

            # -----------------------------------------------------------
            # Confirmation section
            # -----------------------------------------------------------

            if confirmation_active:

                st.warning(
                    f"Are you sure you want to permanently delete "
                    f"**{st.session_state.confirm_delete_filename}**?\n\n"
                    "This will delete the document, its chunks, "
                    "and its extracted images."
                )

                confirm_col, cancel_col = st.columns(2)

                # -------------------------------------------------------
                # Confirm deletion
                # -------------------------------------------------------

                with confirm_col:

                    if st.button(
                        "Confirm Delete",
                        use_container_width=True,
                    ):

                        doc_id_to_delete = (
                            st.session_state.confirm_delete_doc_id
                        )

                        filename_to_delete = (
                            st.session_state.confirm_delete_filename
                        )

                        try:

                            with st.spinner(
                                "Deleting document..."
                            ):

                                response = requests.delete(
                                    f"{API_BASE_URL}/api/v1/upload/"
                                    f"{doc_id_to_delete}",
                                    timeout=120,
                                )

                            if response.ok:

                                st.session_state.confirm_delete_doc_id = None
                                st.session_state.confirm_delete_filename = None

                                st.session_state.delete_message = (
                                    f"Document '{filename_to_delete}' "
                                    "was deleted successfully."
                                )

                                st.rerun()

                            else:

                                st.error(
                                    f"Failed to delete document: "
                                    f"{response.status_code} - "
                                    f"{response.text}"
                                )

                        except requests.RequestException as exc:

                            st.error(
                                "Could not connect to the "
                                f"FastAPI server: {exc}"
                            )

                # -------------------------------------------------------
                # Cancel deletion
                # -------------------------------------------------------

                with cancel_col:

                    if st.button(
                        "Cancel",
                        use_container_width=True,
                    ):

                        st.session_state.confirm_delete_doc_id = None
                        st.session_state.confirm_delete_filename = None

                        st.rerun()

        else:

            st.info(
                "No documents are currently ingested."
            )

        st.divider()

        # ================================================================
        # Clear All Ingested Data
        # ================================================================

        st.subheader("Clear Knowledge Base")

        st.warning(
            "Clearing the knowledge base will delete all "
            "ingested documents, chunks and extracted images."
        )

        # ---------------------------------------------------------------
        # Initial Clear All button
        # ---------------------------------------------------------------

        if not st.session_state.confirm_clear_all:

            if st.button(
                "Clear All Ingested Data",
                use_container_width=True,
            ):

                st.session_state.confirm_clear_all = True

                st.rerun()

        # ---------------------------------------------------------------
        # Clear All confirmation
        # ---------------------------------------------------------------

        if st.session_state.confirm_clear_all:

            st.error(
                "⚠️ This action cannot be undone.\n\n"
                "Are you sure you want to permanently delete "
                "**ALL ingested documents, chunks and extracted images**?"
            )

            clear_confirm_col, clear_cancel_col = st.columns(2)

            # -----------------------------------------------------------
            # Confirm Clear All
            # -----------------------------------------------------------

            with clear_confirm_col:

                if st.button(
                    "Confirm Clear All",
                    use_container_width=True,
                ):

                    try:

                        with st.spinner(
                            "Clearing ingested data..."
                        ):

                            response = requests.delete(
                                f"{API_BASE_URL}/api/v1/upload/clear",
                                timeout=120,
                            )

                        if response.ok:

                            # ------------------------------------------------
                            # Clear confirmation state
                            # ------------------------------------------------

                            st.session_state.confirm_clear_all = False

                            # ------------------------------------------------
                            # Clear individual deletion state
                            # ------------------------------------------------

                            st.session_state.confirm_delete_doc_id = None
                            st.session_state.confirm_delete_filename = None

                            # ------------------------------------------------
                            # Persist clear success message
                            # ------------------------------------------------

                            st.session_state.clear_message = (
                                "All ingested data has been cleared successfully."
                            )

                            st.rerun()

                        else:

                            st.session_state.confirm_clear_all = False

                            st.error(
                                f"Failed to clear data: "
                                f"{response.status_code} - "
                                f"{response.text}"
                            )

                    except requests.RequestException as exc:

                        st.session_state.confirm_clear_all = False

                        st.error(
                            "Could not connect to the "
                            f"FastAPI server: {exc}"
                        )

            # -----------------------------------------------------------
            # Cancel Clear All
            # -----------------------------------------------------------

            with clear_cancel_col:

                if st.button(
                    "Cancel",
                    use_container_width=True,
                ):

                    st.session_state.confirm_clear_all = False

                    st.rerun()

# ---------------------------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

if query := st.chat_input(
    "Ask your credit card question..."
):

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

        error_message = (
            f"Could not connect to the FastAPI server: {exc}"
        )

        with st.chat_message("assistant"):

            st.error(error_message)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )