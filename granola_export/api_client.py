"""
Granola API client for direct server access.

Fetches data from Granola's API endpoints, useful for:
- Shared documents (not in local cache)
- Team-wide exports
- Fresh data without waiting for cache sync
"""

import gzip
import json
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API access."""

    access_token: str
    refresh_token: Optional[str] = None
    client_id: Optional[str] = None
    base_url: str = "https://api.granola.ai"
    user_agent: str = "Granola/5.354.0"
    client_version: str = "5.354.0"


def get_token_from_local() -> Optional[APIConfig]:
    """
    Extract API token from Granola's local storage.

    Returns:
        APIConfig if token found, None otherwise.
    """
    system = platform.system()

    if system == "Darwin":
        token_path = Path.home() / "Library/Application Support/Granola/supabase.json"
    elif system == "Windows":
        token_path = Path.home() / "AppData/Roaming/Granola/supabase.json"
    else:
        token_path = Path.home() / ".config/Granola/supabase.json"

    if not token_path.exists():
        logger.warning(f"Token file not found at {token_path}")
        return None

    try:
        with open(token_path, "r") as f:
            data = json.load(f)

        # Token is in workos_tokens.access_token
        # workos_tokens may be a JSON string or a dict
        workos = data.get("workos_tokens", {})
        if isinstance(workos, str):
            workos = json.loads(workos)

        access_token = workos.get("access_token")

        if not access_token:
            logger.warning("No access_token found in supabase.json")
            return None

        return APIConfig(
            access_token=access_token,
            refresh_token=workos.get("refresh_token"),
        )

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read token file: {e}")
        return None


class GranolaAPIClient:
    """
    Client for Granola's REST API.

    Provides methods to fetch documents, transcripts, workspaces,
    and folders directly from Granola's servers.

    Example:
        >>> client = GranolaAPIClient.from_local_token()
        >>> for doc in client.get_all_documents():
        ...     print(doc['title'])
    """

    def __init__(self, config: APIConfig):
        """
        Initialize the API client.

        Args:
            config: API configuration with access token.
        """
        self.config = config
        self._headers = {
            "Authorization": f"Bearer {config.access_token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": config.user_agent,
            "X-Client-Version": config.client_version,
        }

    @classmethod
    def from_local_token(cls) -> "GranolaAPIClient":
        """
        Create a client using the locally stored token.

        Returns:
            GranolaAPIClient instance.

        Raises:
            ValueError: If no token is found.
        """
        config = get_token_from_local()
        if not config:
            raise ValueError(
                "Could not find Granola API token. "
                "Make sure Granola is installed and you're logged in."
            )
        return cls(config)

    @classmethod
    def from_token(cls, access_token: str) -> "GranolaAPIClient":
        """
        Create a client with an explicit token.

        Args:
            access_token: The API access token.

        Returns:
            GranolaAPIClient instance.
        """
        return cls(APIConfig(access_token=access_token))

    def _request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict:
        """Make an API request with retry logic.

        Args:
            endpoint: API endpoint path.
            method: HTTP method.
            data: Request payload.
            max_retries: Maximum retry attempts for transient failures.

        Returns:
            Parsed JSON response.

        Raises:
            urllib.error.HTTPError: For non-retryable HTTP errors.
            urllib.error.URLError: After all retries exhausted.
        """
        url = f"{self.config.base_url}{endpoint}"
        body = json.dumps(data or {}).encode("utf-8")

        last_exception = None

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                headers=self._headers,
                method=method,
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    raw_data = response.read()

                    # Check if response is gzip-compressed
                    if raw_data[:2] == b'\x1f\x8b':
                        raw_data = gzip.decompress(raw_data)

                    return json.loads(raw_data.decode("utf-8"))

            except urllib.error.HTTPError as e:
                # Retry on 5xx server errors
                if e.code >= 500 and attempt < max_retries:
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Server error {e.code}, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    last_exception = e
                    continue

                # Non-retryable HTTP error
                error_body = ""
                if e.fp:
                    raw = e.read()
                    if raw[:2] == b'\x1f\x8b':
                        raw = gzip.decompress(raw)
                    error_body = raw.decode("utf-8")
                logger.error(f"API error {e.code}: {error_body}")
                raise

            except urllib.error.URLError as e:
                # Retry on network errors (includes SSL timeouts)
                if attempt < max_retries:
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Network error: {e.reason}, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    last_exception = e
                    continue

                logger.error(f"Network error after {max_retries} retries: {e.reason}")
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------

    def get_documents(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """
        Fetch documents from the API.

        Note: This endpoint only returns OWNED documents, not shared ones.
        Use get_documents_batch() for shared documents.

        Args:
            workspace_id: Optional workspace filter.
            limit: Max documents per request.
            offset: Pagination offset.

        Returns:
            API response with 'docs' array.
        """
        data = {
            "limit": limit,
            "offset": offset,
            "include_last_viewed_panel": True,
        }

        if workspace_id:
            data["workspace_id"] = workspace_id

        return self._request("/v2/get-documents", data=data)

    def get_all_documents(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> Iterator[dict]:
        """
        Iterate through all documents with automatic pagination.

        Args:
            workspace_id: Optional workspace filter.
            limit: Batch size.

        Yields:
            Document dictionaries.
        """
        offset = 0

        while True:
            response = self.get_documents(
                workspace_id=workspace_id,
                limit=limit,
                offset=offset,
            )

            docs = response.get("docs", [])
            if not docs:
                break

            for doc in docs:
                yield doc

            if len(docs) < limit:
                break

            offset += limit

    def get_documents_batch(
        self,
        document_ids: list[str],
        batch_size: int = 100,
    ) -> list[dict]:
        """
        Fetch multiple documents by ID, including shared documents.

        This is the ONLY way to fetch shared documents that aren't
        returned by get_documents().

        Args:
            document_ids: List of document IDs.
            batch_size: Documents per request.

        Returns:
            List of document dictionaries.
        """
        all_docs = []

        for i in range(0, len(document_ids), batch_size):
            batch = document_ids[i : i + batch_size]
            data = {
                "document_ids": batch,
                "include_last_viewed_panel": True,
            }

            response = self._request("/v1/get-documents-batch", data=data)
            docs = response.get("documents") or response.get("docs") or []
            all_docs.extend(docs)

        return all_docs

    # -------------------------------------------------------------------------
    # Transcripts
    # -------------------------------------------------------------------------

    def get_document_transcript(self, document_id: str) -> Optional[list]:
        """
        Fetch transcript for a specific document.

        Args:
            document_id: The document ID.

        Returns:
            List of transcript segments, or None if not found.
        """
        try:
            response = self._request(
                "/v1/get-document-transcript",
                data={"document_id": document_id},
            )
            # API returns transcript array directly or in 'transcript' key
            if isinstance(response, list):
                return response
            return response.get("transcript", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    # -------------------------------------------------------------------------
    # Workspaces and Folders
    # -------------------------------------------------------------------------

    def get_workspaces(self) -> list[dict]:
        """
        Fetch all accessible workspaces.

        Returns:
            List of workspace dictionaries.
        """
        response = self._request("/v1/get-workspaces", data={})

        if isinstance(response, list):
            return response
        return response.get("workspaces", [])

    def get_document_lists(self) -> list[dict]:
        """
        Fetch all document lists (folders).

        Returns:
            List of folder dictionaries with document IDs.
        """
        # Try v2 first, fall back to v1
        try:
            response = self._request("/v2/get-document-lists", data={})
        except urllib.error.HTTPError:
            response = self._request("/v1/get-document-lists", data={})

        if isinstance(response, list):
            return response

        return (
            response.get("lists")
            or response.get("document_lists")
            or []
        )

    # -------------------------------------------------------------------------
    # Other Endpoints
    # -------------------------------------------------------------------------

    def get_people(self) -> dict:
        """Fetch people/contacts data."""
        return self._request("/v1/get-people", data={})

    def get_panel_templates(self) -> list[dict]:
        """Fetch available panel templates."""
        response = self._request("/v1/get-panel-templates", data={})
        if isinstance(response, list):
            return response
        return response.get("panel_templates", [])

    def get_feature_flags(self) -> dict:
        """Fetch feature flags."""
        return self._request("/v1/get-feature-flags", data={})

    def check_connection(self) -> bool:
        """
        Test if the API connection is working.

        Returns:
            True if connection successful.
        """
        try:
            self.get_workspaces()
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
