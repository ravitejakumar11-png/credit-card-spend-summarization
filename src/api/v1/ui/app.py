import json
import re
import uuid

import requests
import streamlit as st

# ===========================================================================
# Configuration
# ===========================================================================

API_BASE_URL = "http://127.0.0.1:8000"

# ===========================================================================
# Page configuration
# ===========================================================================

st.set_page_config(
    page_title="NorthStar Credit Card Assistant",
    page_icon="💳",
    layout="wide",
)

st.title("NorthStar Credit Card Assistant")

# ===========================================================================
# Session State
# ===========================================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

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


# ===========================================================================
# Helper: Extract final user-facing answer
# ===========================================================================


def extract_final_answer(full_stream: str) -> str:
    """
    Extract ONLY the user-facing answer from the complete LLM stream.

    Supported formats:

    1. HYBRID / RDBMS / VECTOR
       {"answer": "James Mitchell has spent Rs. 232,500.00 ..."}

    2. DIRECT - Pydantic representation
       RouteDecision(
           route='DIRECT',
           reason='...',
           direct_response='Hello John! How can I assist you today?'
       )

    3. DIRECT - JSON representation
       {
           "route": "DIRECT",
           "reason": "...",
           "direct_response": "Hello John!"
       }

    The function deliberately ignores route, reason, SQL, document_name,
    page_no, policy_citations, sql_query_executed, evaluation, and
    evaluation_feedback.
    """

    if not full_stream:
        return ""

    if not isinstance(full_stream, str):
        full_stream = str(full_stream)

    # =========================================================================
    # 1. Normal final `answer` field
    # =========================================================================

    answer_match = re.search(
        r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',
        full_stream,
        flags=re.DOTALL,
    )

    if answer_match:
        encoded_answer = answer_match.group(1)

        try:
            answer = json.loads('"' + encoded_answer + '"')

            if answer:
                return answer.strip()

        except json.JSONDecodeError:
            answer = (
                encoded_answer.replace("\\n", "\n")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
                .strip()
            )

            if answer:
                return answer

    # =========================================================================
    # 2. DIRECT RouteDecision / Pydantic representation
    # =========================================================================

    direct_match = re.search(
        r"direct_response\s*=\s*(['\"])(.*?)\1",
        full_stream,
        flags=re.DOTALL,
    )

    if direct_match:
        direct_response = direct_match.group(2).strip()

        if direct_response:
            return direct_response

    # =========================================================================
    # 3. DIRECT JSON format
    # =========================================================================

    direct_json_match = re.search(
        r'"direct_response"\s*:\s*"((?:\\.|[^"\\])*)"',
        full_stream,
        flags=re.DOTALL,
    )

    if direct_json_match:
        encoded_response = direct_json_match.group(1)

        try:
            direct_response = json.loads('"' + encoded_response + '"')

            if direct_response:
                return direct_response.strip()

        except json.JSONDecodeError:
            direct_response = (
                encoded_response.replace("\\n", "\n")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
                .strip()
            )

            if direct_response:
                return direct_response

    # =========================================================================
    # 4. Nothing user-facing found
    # =========================================================================

    return ""


with st.sidebar:

    st.header("Developer Mode")

    developer_mode = st.toggle(
        "Enable Developer Mode",
        value=False,
    )

    # -----------------------------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------------------------

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

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

            st.success(st.session_state.ingest_message)

            if st.session_state.ingest_result:

                st.json(st.session_state.ingest_result)

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

                    # ---------------------------------------------------
                    # Persist ingestion response before rerun.
                    # ---------------------------------------------------

                    st.session_state.ingest_message = "Document ingested successfully."

                    st.session_state.ingest_result = result

                    st.rerun()

                else:

                    st.error(
                        f"Ingestion failed: "
                        f"{response.status_code} - "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:

                st.error(f"Could not connect to the FastAPI server: {exc}")

        st.divider()

        # ================================================================
        # Knowledge Base
        # ================================================================

        st.subheader("Knowledge Base")

        # ---------------------------------------------------------------
        # Display individual document deletion message
        # ---------------------------------------------------------------

        if st.session_state.delete_message:

            st.success(st.session_state.delete_message)

            st.session_state.delete_message = None

        # ---------------------------------------------------------------
        # Display clear-all success message
        # ---------------------------------------------------------------

        if st.session_state.clear_message:

            st.success(st.session_state.clear_message)

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

            st.error(f"Could not connect to the FastAPI server: {exc}")

        # ---------------------------------------------------------------
        # Delete individual document
        # ---------------------------------------------------------------

        st.markdown("**Delete Document**")

        if documents:

            document_options = {
                document["filename"]: document for document in documents
            }

            selected_filename = st.selectbox(
                "Select document",
                options=list(document_options.keys()),
            )

            selected_document = document_options[selected_filename]

            selected_doc_id = str(selected_document["id"])

            st.caption(f"Document ID: {selected_doc_id}")

            # -----------------------------------------------------------
            # Check whether this document is awaiting confirmation
            # -----------------------------------------------------------

            confirmation_active = (
                st.session_state.confirm_delete_doc_id == selected_doc_id
            )

            # -----------------------------------------------------------
            # Normal delete button
            # -----------------------------------------------------------

            if not confirmation_active:

                if st.button(
                    "Delete Selected Document",
                    use_container_width=True,
                ):

                    st.session_state.confirm_delete_doc_id = selected_doc_id

                    st.session_state.confirm_delete_filename = selected_filename

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

                        doc_id_to_delete = st.session_state.confirm_delete_doc_id

                        filename_to_delete = st.session_state.confirm_delete_filename

                        try:

                            with st.spinner("Deleting document..."):

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
                                "Could not connect to the " f"FastAPI server: {exc}"
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

            st.info("No documents are currently ingested.")

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

                        with st.spinner("Clearing ingested data..."):

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

                        st.error("Could not connect to the " f"FastAPI server: {exc}")

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

# ===========================================================================
# Display previous chat messages
# ===========================================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ===========================================================================
# User Input
# ===========================================================================

query = st.chat_input("Ask your credit card question...")


if query:

    # =========================================================================
    # Display User Message
    # =========================================================================

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # =========================================================================
    # Assistant Message
    # =========================================================================

    with st.chat_message("assistant"):

        answer_placeholder = st.empty()

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT display tokens immediately.
        #
        # Your backend streams ALL LLM output:
        #
        #   route
        #   reason
        #   SQL
        #   answer
        #   evaluation
        #
        # Therefore we collect the entire stream first.
        # ---------------------------------------------------------------------

        full_stream = ""

        try:

            # =================================================================
            # Call FastAPI streaming endpoint
            # =================================================================

            response = requests.post(
                f"{API_BASE_URL}/api/v1/query/stream/",
                json={
                    "query": query,
                    "thread_id": st.session_state.thread_id,
                },
                stream=True,
                timeout=300,
            )

            # =================================================================
            # HTTP Error
            # =================================================================

            if not response.ok:

                error_message = (
                    f"API request failed: "
                    f"{response.status_code} - "
                    f"{response.text}"
                )

                answer_placeholder.error(error_message)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

            else:

                # =============================================================
                # Check response type
                # =============================================================

                content_type = response.headers.get(
                    "content-type",
                    "",
                ).lower()

                # =============================================================
                # SSE Response
                # =============================================================

                if "text/event-stream" in content_type:

                    for line in response.iter_lines(decode_unicode=True):

                        # -----------------------------------------------------
                        # Ignore empty SSE lines
                        # -----------------------------------------------------

                        if not line:
                            continue

                        # -----------------------------------------------------
                        # We only process:
                        #
                        # data: ...
                        # -----------------------------------------------------

                        if not line.startswith("data:"):
                            continue

                        # -----------------------------------------------------
                        # Remove "data:" prefix
                        # -----------------------------------------------------

                        data = line[len("data:") :].strip()

                        # -----------------------------------------------------
                        # End of stream
                        # -----------------------------------------------------

                        if data == "[DONE]":
                            break

                        # -----------------------------------------------------
                        # Parse SSE JSON
                        #
                        # Backend sends:
                        #
                        # {
                        #     "token": "..."
                        # }
                        # -----------------------------------------------------

                        try:

                            chunk = json.loads(data)

                        except json.JSONDecodeError:

                            # Ignore malformed events.
                            continue

                        # -----------------------------------------------------
                        # Make sure the payload is an object.
                        # -----------------------------------------------------

                        if not isinstance(
                            chunk,
                            dict,
                        ):
                            continue

                        # -----------------------------------------------------
                        # Extract token
                        # -----------------------------------------------------

                        token = chunk.get(
                            "token",
                            "",
                        )

                        # -----------------------------------------------------
                        # Only collect string tokens.
                        #
                        # This also protects Streamlit from unexpected
                        # structured callback values.
                        # -----------------------------------------------------

                        if isinstance(
                            token,
                            str,
                        ):

                            full_stream += token

                # =============================================================
                # Non-SSE fallback
                # =============================================================

                else:

                    full_stream = response.text

                # =============================================================
                # IMPORTANT:
                #
                # Stream has now completed.
                #
                # Extract ONLY the user-facing response.
                # =============================================================

                answer = extract_final_answer(full_stream)

                # =============================================================
                # Display extracted answer
                # =============================================================

                if answer:

                    answer_placeholder.markdown(answer)

                    # ---------------------------------------------------------
                    # Save ONLY answer in chat history.
                    #
                    # We deliberately DO NOT save full_stream.
                    # ---------------------------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                # =============================================================
                # No answer found
                # =============================================================

                else:

                    answer_placeholder.warning(
                        "The server completed the stream "
                        "without returning a user-facing answer."
                    )

                    # ---------------------------------------------------------
                    # Developer debugging
                    #
                    # This is the ONLY place where the raw stream can be
                    # displayed, and only when Developer Mode is enabled.
                    # ---------------------------------------------------------

                    if developer_mode:

                        st.divider()

                        st.subheader("Raw Stream Debug")

                        st.code(full_stream)

        # =====================================================================
        # Connection Error
        # =====================================================================

        except requests.RequestException as exc:

            error_message = "Could not connect to the FastAPI server: " f"{exc}"

            answer_placeholder.error(error_message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
