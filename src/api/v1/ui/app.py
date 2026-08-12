import json
import uuid
import queue

from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://127.0.0.1:8000"

_QUERY_EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="NorthStar Credit Card Assistant",
    page_icon="💳",
    layout="wide",
)

st.title("NorthStar Credit Card Assistant")


# ============================================================================
# Conversation thread
# ============================================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ============================================================================
# Chat history
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================================
# Query execution state
# ============================================================================

if "query_running" not in st.session_state:
    st.session_state.query_running = False

if "active_query" not in st.session_state:
    st.session_state.active_query = None

if "query_future" not in st.session_state:
    st.session_state.query_future = None

if "query_cancel_requested" not in st.session_state:
    st.session_state.query_cancel_requested = False

if "query_error" not in st.session_state:
    st.session_state.query_error = None

if "query_progress" not in st.session_state:
    st.session_state.query_progress = None

if "query_progress_queue" not in st.session_state:
    st.session_state.query_progress_queue = None
# ============================================================================
# Delete / Clear / Ingestion state
# ============================================================================

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


# ============================================================================
# Ingestion result state
# ============================================================================

if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = None

if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None


# ============================================================================
# Background query worker
# ============================================================================


def _execute_query_request(
    query: str,
    thread_id: str,
    progress_queue: queue.Queue,
) -> dict:

    print("========== BACKGROUND QUERY WORKER ==========")

    print(f"Query     : {query!r}")

    print(f"Thread ID : {thread_id!r}")

    try:

        with requests.post(
            f"{API_BASE_URL}/api/v1/query/stream",
            json={
                "query": query,
                "thread_id": thread_id,
            },
            stream=True,
            timeout=300,
        ) as response:

            print("========== STREAMING API RESPONSE ==========")

            print(f"Status code : {response.status_code}")

            if not response.ok:

                return {
                    "status": "error",
                    "error": (
                        f"API request failed: "
                        f"{response.status_code} - "
                        f"{response.text}"
                    ),
                }

            # IMPORTANT:
            # chunk_size=1 helps small NDJSON progress events arrive
            # without unnecessary buffering.
            for line in response.iter_lines(
                chunk_size=1,
                decode_unicode=True,
            ):

                if not line:
                    continue

                try:

                    event = json.loads(line)

                except json.JSONDecodeError:

                    print("[query_worker] " "Ignoring invalid stream event.")

                    continue

                event_type = event.get("event")

                # ============================================================
                # PROGRESS EVENT
                # ============================================================

                if event_type == "progress":

                    message = event.get(
                        "message",
                        "",
                    )

                    if message:

                        print("[query_worker] Progress: " f"{message}")

                        # Send progress to the Streamlit fragment.
                        progress_queue.put(message)

                    continue

                # ============================================================
                # FINAL RESPONSE
                # ============================================================

                if event_type == "final":

                    return {
                        "status": "success",
                        "response": event.get("data"),
                    }

                # ============================================================
                # CANCELLATION
                # ============================================================

                if event_type == "cancelled":

                    return {
                        "status": "cancelled",
                        "message": event.get(
                            "message",
                            "The query was stopped.",
                        ),
                    }

                # ============================================================
                # BACKEND ERROR
                # ============================================================

                if event_type == "error":

                    return {
                        "status": "error",
                        "error": event.get(
                            "message",
                            "Unable to process your question.",
                        ),
                    }

            return {
                "status": "error",
                "error": (
                    "The server ended the request "
                    "without returning a final response."
                ),
            }

    except requests.RequestException as exc:

        print("[query_worker] Request failed: " f"{exc}")

        return {
            "status": "error",
            "error": ("Could not connect to the FastAPI server. " "Please try again."),
        }

    except Exception as exc:

        print("[query_worker] Unexpected error: " f"{exc}")

        return {
            "status": "error",
            "error": (
                "Something went wrong while processing "
                "your question. Please try again."
            ),
        }


# ============================================================================
# Developer Mode Sidebar
# ============================================================================

with st.sidebar:

    st.header("Developer Mode")

    developer_mode = st.toggle(
        "Enable Developer Mode",
        value=False,
    )

    if developer_mode:

        st.divider()

        # ====================================================================
        # Document Ingestion
        # ====================================================================

        st.subheader("Document Ingestion")

        if st.session_state.ingest_message:

            st.success(st.session_state.ingest_message)

            if st.session_state.ingest_result:

                st.json(st.session_state.ingest_result)

            st.session_state.ingest_message = None
            st.session_state.ingest_result = None

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

        # ====================================================================
        # Knowledge Base
        # ====================================================================

        st.subheader("Knowledge Base")

        if st.session_state.delete_message:

            st.success(st.session_state.delete_message)

            st.session_state.delete_message = None

        if st.session_state.clear_message:

            st.success(st.session_state.clear_message)

            st.session_state.clear_message = None

        # --------------------------------------------------------------------
        # Load documents
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # Delete individual document
        # --------------------------------------------------------------------

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

            confirmation_active = (
                st.session_state.confirm_delete_doc_id == selected_doc_id
            )

            if not confirmation_active:

                if st.button(
                    "Delete Selected Document",
                    use_container_width=True,
                ):

                    st.session_state.confirm_delete_doc_id = selected_doc_id

                    st.session_state.confirm_delete_filename = selected_filename

                    st.rerun()

            if confirmation_active:

                st.warning(
                    f"Are you sure you want to permanently delete "
                    f"**{st.session_state.confirm_delete_filename}**?\n\n"
                    "This will delete the document, its chunks, "
                    "and its extracted images."
                )

                confirm_col, cancel_col = st.columns(2)

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

        # ====================================================================
        # Clear All
        # ====================================================================

        st.subheader("Clear Knowledge Base")

        st.warning(
            "Clearing the knowledge base will delete all "
            "ingested documents, chunks and extracted images."
        )

        if not st.session_state.confirm_clear_all:

            if st.button(
                "Clear All Ingested Data",
                use_container_width=True,
            ):

                st.session_state.confirm_clear_all = True

                st.rerun()

        if st.session_state.confirm_clear_all:

            st.error(
                "⚠️ This action cannot be undone.\n\n"
                "Are you sure you want to permanently delete "
                "**ALL ingested documents, chunks and extracted images**?"
            )

            clear_confirm_col, clear_cancel_col = st.columns(2)

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

                            st.session_state.confirm_clear_all = False

                            st.session_state.confirm_delete_doc_id = None
                            st.session_state.confirm_delete_filename = None

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
                            "Could not connect to the " f"{API_BASE_URL} server: {exc}"
                        )

            with clear_cancel_col:

                if st.button(
                    "Cancel",
                    use_container_width=True,
                ):

                    st.session_state.confirm_clear_all = False

                    st.rerun()


# ============================================================================
# Display chat history
#
# IMPORTANT:
# The final answer is persisted here BEFORE the full rerun.
# This is what prevents the answer from disappearing.
# ============================================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================================
# Active query fragment
#
# IMPORTANT:
# This fragment NEVER permanently renders the final answer.
#
# Its only responsibilities are:
#
#   1. Show that processing is happening.
#   2. Show the Stop button.
#   3. Monitor the background request.
#   4. Persist the final result.
#   5. Trigger a full rerun.
#
# The normal chat-history section above then displays the answer.
# ============================================================================


@st.fragment(run_every="1s")
def active_query_fragment():

    if not st.session_state.query_running:

        return

    # ------------------------------------------------------------------------
    # Read progress events produced by the background query worker.
    # ------------------------------------------------------------------------

    progress_queue = st.session_state.query_progress_queue

    if progress_queue is not None:

        latest_progress = None

        while True:

            try:

                latest_progress = progress_queue.get_nowait()

            except queue.Empty:

                break

        if latest_progress:

            st.session_state.query_progress = latest_progress
    # ------------------------------------------------------------------------
    # Processing indicator
    # ------------------------------------------------------------------------

    if st.session_state.query_cancel_requested:

        st.info("Stopping your request...")

    else:

        progress_message = (
            st.session_state.query_progress or "Processing your question..."
        )

        st.info(progress_message)

    # ------------------------------------------------------------------------
    # Stop button
    # ------------------------------------------------------------------------

    if st.session_state.query_cancel_requested:

        st.button(
            "⏹ Stopping...",
            disabled=True,
            use_container_width=True,
        )

    else:

        if st.button(
            "⏹ Stop",
            use_container_width=True,
        ):

            print("========== QUERY CANCELLATION REQUEST ==========")

            print(f"Thread ID : " f"{st.session_state.thread_id!r}")

            try:

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/query/cancel/"
                    f"{st.session_state.thread_id}",
                    timeout=10,
                )

                if response.ok:

                    print("[ui] Cancellation requested.")

                    st.session_state.query_cancel_requested = True

                    st.rerun()

                else:

                    st.error(
                        "Unable to stop the query: "
                        f"{response.status_code} - "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:

                st.error("Could not connect to the " f"{API_BASE_URL} server: {exc}")

    # ------------------------------------------------------------------------
    # Get background future
    # ------------------------------------------------------------------------

    future = st.session_state.query_future

    if future is None:

        st.session_state.query_running = False

        st.session_state.query_error = "The query could not be started."

        st.rerun()

    # ------------------------------------------------------------------------
    # Query is still running
    # ------------------------------------------------------------------------

    if not future.done():

        return

    # ------------------------------------------------------------------------
    # Query has completed
    # ------------------------------------------------------------------------

    try:

        result = future.result()

    except Exception as exc:

        print("[ui] Background query failed: " f"{exc}")

        result = {
            "status": "error",
            "error": (
                "Something went wrong while processing "
                "your question. Please try again."
            ),
        }

    # ------------------------------------------------------------------------
    # Capture cancellation state BEFORE resetting it.
    # ------------------------------------------------------------------------

    was_cancel_requested = st.session_state.query_cancel_requested

    # ------------------------------------------------------------------------
    # Clear active query state.
    # ------------------------------------------------------------------------

    st.session_state.query_running = False

    st.session_state.query_future = None

    st.session_state.active_query = None

    st.session_state.query_cancel_requested = False

    st.session_state.query_progress = None

    st.session_state.query_progress_queue = None

    # ------------------------------------------------------------------------
    # If user pressed Stop, do not display the eventual answer.
    # ------------------------------------------------------------------------

    if was_cancel_requested:

        stop_message = "The query was stopped."

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": stop_message,
            }
        )

        # Full rerun:
        #
        # - active fragment disappears
        # - chat input becomes enabled
        # - stop message appears in chat history

        st.rerun()

    # ------------------------------------------------------------------------
    # Successful response
    # ------------------------------------------------------------------------

    if result.get("status") == "success":

        final_response = result.get("response")

        if final_response:

            answer = final_response.get(
                "answer",
                "",
            )

            if answer:

                # ------------------------------------------------------------
                # IMPORTANT:
                #
                # Persist the answer BEFORE rerunning.
                # ------------------------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            else:

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            "The assistant could not produce "
                            "a response. Please try again."
                        ),
                    }
                )

        else:

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "The assistant did not return a response. " "Please try again."
                    ),
                }
            )

        # --------------------------------------------------------------------
        # Full rerun.
        #
        # This is critical:
        #
        # - answer is now in chat_history
        # - query_running is False
        # - chat input becomes enabled
        # - normal chat rendering displays the answer
        # --------------------------------------------------------------------

        st.rerun()

    # ------------------------------------------------------------------------
    # Backend/API error
    # ------------------------------------------------------------------------

    error_message = result.get(
        "error",
        "Unable to process your question.",
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": error_message,
        }
    )

    st.rerun()


# ============================================================================
# Run active query monitoring
# ============================================================================

active_query_fragment()


# ============================================================================
# User input
#
# It is disabled ONLY while an actual query is running.
#
# Once the fragment stores the answer and calls st.rerun(),
# query_running becomes False and this becomes active again.
# ============================================================================

query = st.chat_input(
    "Ask your credit card question...",
    disabled=st.session_state.query_running,
)


# ============================================================================
# Start new query
# ============================================================================

if query and not st.session_state.query_running:

    print("========== STREAMLIT QUERY ==========")

    print(f"Query     : {query!r}")

    print(f"Thread ID : " f"{st.session_state.thread_id!r}")

    # ------------------------------------------------------------------------
    # Persist user message immediately.
    # ------------------------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # ------------------------------------------------------------------------
    # Initialize query state.
    # ------------------------------------------------------------------------

    st.session_state.active_query = query

    st.session_state.query_running = True

    st.session_state.query_cancel_requested = False

    st.session_state.query_error = None

    st.session_state.query_progress = "Processing your question..."

    # ------------------------------------------------------------------------
    # Start background request.
    # ------------------------------------------------------------------------

    st.session_state.query_progress_queue = queue.Queue()

    st.session_state.query_future = _QUERY_EXECUTOR.submit(
        _execute_query_request,
        query,
        st.session_state.thread_id,
        st.session_state.query_progress_queue,
    )
    # ------------------------------------------------------------------------
    # Rerun immediately.
    #
    # The active-query fragment will now show:
    #
    #     Processing your question...
    #     [ Stop ]
    #
    # while the background request continues.
    # ------------------------------------------------------------------------

    st.rerun()
