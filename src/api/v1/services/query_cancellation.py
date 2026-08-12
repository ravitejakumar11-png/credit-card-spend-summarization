import threading

_cancelled_threads: set[str] = set()
_lock = threading.Lock()


class QueryCancelled(Exception):
    """Raised when a user requests cancellation of a running query."""

    pass


def cancel_query(thread_id: str) -> None:
    with _lock:
        _cancelled_threads.add(thread_id)

    print(f"[cancellation] Cancellation requested: {thread_id}")


def is_query_cancelled(thread_id: str | None) -> bool:
    if not thread_id:
        return False

    with _lock:
        return thread_id in _cancelled_threads


def clear_query_cancellation(thread_id: str | None) -> None:
    if not thread_id:
        return

    with _lock:
        _cancelled_threads.discard(thread_id)

    print(f"[cancellation] Cancellation state cleared: {thread_id}")


def raise_if_query_cancelled(thread_id: str | None) -> None:
    if is_query_cancelled(thread_id):
        print(f"[cancellation] Query cancelled: {thread_id}")
        raise QueryCancelled()
