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
# Session helpers
# ============================================================================


def _start_new_thread() -> None:
    """Start a clean conversation thread."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []


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

if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ============================================================================
# User Login Gate
# ============================================================================

# Login happens before the chat UI is displayed.
# User preferences are loaded only after Login/Guest selection.

if not st.session_state.authenticated:

    st.subheader("Welcome to NorthStar Credit Card Assistant")

    st.write(
        "Please login with your User ID to enable personalized preferences, "
        "or continue as a guest."
    )

    login_user_id = st.text_input(
        "User ID",
        placeholder="e.g. C-1001",
        help=(
            "Enter a user ID to enable persistent Mem0 preferences. "
            "No username/password validation is performed."
        ),
    )

    login_col, guest_col = st.columns(2)

    with login_col:
        if st.button("Login", use_container_width=True):
            login_user_id = login_user_id.strip()

            if login_user_id:
                st.session_state.user_id = login_user_id
                st.session_state.authenticated = True
                _start_new_thread()
                st.rerun()
            else:
                st.warning("Please enter a User ID.")

    with guest_col:
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.user_id = ""
            st.session_state.authenticated = True
            _start_new_thread()
            st.rerun()

    st.caption(
        "Guest requests use an empty user_id. " "Login accepts any non-empty User ID."
    )

    st.stop()


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

if "streaming_answer" not in st.session_state:
    st.session_state.streaming_answer = ""


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
    user_id: str,
    progress_queue: queue.Queue,
) -> dict:
    """Run the FastAPI streaming request in a background thread.

    The backend sends SSE events for progress, answer tokens, reset events,
    final response, cancellation, and errors. Events are forwarded to the
    Streamlit fragment through a thread-safe queue.
    """

    print("========== BACKGROUND QUERY WORKER ==========")
    print(f"Query     : {query!r}")
    print(f"Thread ID : {thread_id!r}")
    print(f"User ID   : {user_id!r}")

    try:
        with requests.post(
            f"{API_BASE_URL}/api/v1/query/stream",
            json={
                "query": query,
                "thread_id": thread_id,
                "user_id": user_id,
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

            for line in response.iter_lines(
                chunk_size=1,
                decode_unicode=True,
            ):
                if not line:
                    continue

                # FastAPI returns Server-Sent Events in the form:
                # data: {JSON payload}
                if not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()

                if data == "[DONE]":
                    break

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    print("[query_worker] Ignoring invalid stream event.")
                    continue

                if not isinstance(event, dict):
                    continue

                event_type = event.get("event")

                if event_type == "progress":
                    message = event.get("message", "")
                    if message:
                        print(f"[query_worker] Progress: {message}")
                        progress_queue.put(
                            {
                                "event": "progress",
                                "message": message,
                            }
                        )
                    continue

                if event_type == "token":
                    token = event.get("content", "")
                    if token:
                        progress_queue.put(
                            {
                                "event": "token",
                                "content": token,
                            }
                        )
                    continue

                if event_type == "reset":
                    # The evaluator may reject an answer and request
                    # regeneration. Forward reset so the UI does not
                    # concatenate the rejected answer with the replacement.
                    progress_queue.put(
                        {
                            "event": "reset",
                        }
                    )
                    continue

                if event_type == "final":
                    return {
                        "status": "success",
                        "response": event.get("data"),
                    }

                if event_type == "cancelled":
                    return {
                        "status": "cancelled",
                        "message": event.get(
                            "message",
                            "The query was stopped.",
                        ),
                    }

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
        print(f"[query_worker] Request failed: {exc}")
        return {
            "status": "error",
            "error": ("Could not connect to the FastAPI server. " "Please try again."),
        }

    except Exception as exc:
        print(f"[query_worker] Unexpected error: {exc}")
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

    st.header("User")

    if st.session_state.user_id:
        st.success(f"Logged in as: {st.session_state.user_id}")
    else:
        st.info("Using Guest mode")

    if st.button(
        "Logout",
        use_container_width=True,
        disabled=st.session_state.query_running,
    ):
        st.session_state.user_id = ""
        st.session_state.authenticated = False
        _start_new_thread()
        st.rerun()

    st.divider()

    # ========================================================================
    # st.header("Developer Mode")

    developer_mode = False

    if st.session_state.user_id == "admin":
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
        _start_new_thread()
        st.rerun()

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
        st.markdown(message["content"])


# ============================================================================
# Active query fragment
#
# IMPORTANT:
# This fragment NEVER permanently renders the final answer.
#
# Its responsibilities are:
#
#   1. Show that processing is happening.
#   2. Show the Stop button.
#   3. Monitor the background request.
#   4. Persist the final result.
#   5. Trigger a full rerun.
#
# The normal chat-history section above then displays the answer.
# ============================================================================


@st.fragment(run_every=0.2)
def active_query_fragment():

    if not st.session_state.query_running:
        return

    # ------------------------------------------------------------------------
    # Read progress/token/reset events from the background query worker.
    # ------------------------------------------------------------------------

    progress_queue = st.session_state.query_progress_queue

    if progress_queue is not None:
        latest_progress = None

        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break

            if not isinstance(event, dict):
                continue

            event_type = event.get("event")

            if event_type == "progress":
                message = event.get("message", "")
                if message:
                    latest_progress = message

            elif event_type == "token":
                token = event.get("content", "")
                if token:
                    st.session_state.streaming_answer += token

            elif event_type == "reset":
                # Evaluation may request a regeneration. The backend sends
                # reset before the replacement answer starts so the UI does
                # not concatenate the rejected answer with the new one.
                st.session_state.streaming_answer = ""

        if latest_progress:
            st.session_state.query_progress = latest_progress

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
    #
    # Check future.done() BEFORE rendering the progress message. This
    # prevents a stale processing message from being rendered after the
    # backend has already completed.
    # ------------------------------------------------------------------------

    if not future.done():

        # --------------------------------------------------------------------
        # Processing indicator
        #
        # Once answer tokens have started arriving, hide the progress
        # message. The streamed answer becomes the primary UI.
        # --------------------------------------------------------------------

        if not st.session_state.streaming_answer:

            if st.session_state.query_cancel_requested:
                st.info("Stopping your request...")
            else:
                progress_message = (
                    st.session_state.query_progress or "Processing your question..."
                )
                st.info(progress_message)

        # --------------------------------------------------------------------
        # Live streamed answer
        # --------------------------------------------------------------------

        if st.session_state.streaming_answer:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.streaming_answer)

        # --------------------------------------------------------------------
        # Stop button
        # --------------------------------------------------------------------

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
                print(f"Thread ID : {st.session_state.thread_id!r}")

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
                    st.error(
                        "Could not connect to the " f"{API_BASE_URL} server: {exc}"
                    )

        return

    # ------------------------------------------------------------------------
    # Query has completed
    # ------------------------------------------------------------------------

    try:
        result = future.result()

    except Exception as exc:
        print(f"[ui] Background query failed: {exc}")
        result = {
            "status": "error",
            "error": (
                "Something went wrong while processing "
                "your question. Please try again."
            ),
        }

    # ------------------------------------------------------------------------
    # Drain token/reset events that arrived immediately before completion.
    # This closes the small race between the worker receiving the final SSE
    # event and this fragment observing future.done().
    # ------------------------------------------------------------------------

    if progress_queue is not None:

        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break

            if not isinstance(event, dict):
                continue

            event_type = event.get("event")

            if event_type == "token":
                token = event.get("content", "")
                if token:
                    st.session_state.streaming_answer += token

            elif event_type == "reset":
                st.session_state.streaming_answer = ""

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

    if was_cancel_requested or result.get("status") == "cancelled":

        st.session_state.streaming_answer = ""

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.get(
                    "message",
                    "The query was stopped.",
                ),
            }
        )

        st.rerun()

    # ------------------------------------------------------------------------
    # Successful response
    # ------------------------------------------------------------------------

    if result.get("status") == "success":

        final_response = result.get("response")

        if final_response:

            answer = final_response.get("answer", "")

            if answer:

                # The final structured response is authoritative.
                # Persist it before rerunning so it appears in chat history.
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
                            "The assistant did not return a response. "
                            "Please try again."
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

        st.session_state.streaming_answer = ""
        st.rerun()

    # ------------------------------------------------------------------------
    # Backend/API error
    # ------------------------------------------------------------------------

    error_message = result.get(
        "error",
        "Unable to process your question.",
    )

    st.session_state.streaming_answer = ""

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
    print(f"Thread ID : {st.session_state.thread_id!r}")
    print(f"User ID   : {st.session_state.user_id!r}")

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
    st.session_state.streaming_answer = ""

    # ------------------------------------------------------------------------
    # Start background request.
    # ------------------------------------------------------------------------

    st.session_state.query_progress_queue = queue.Queue()

    st.session_state.query_future = _QUERY_EXECUTOR.submit(
        _execute_query_request,
        query,
        st.session_state.thread_id,
        st.session_state.user_id,
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
