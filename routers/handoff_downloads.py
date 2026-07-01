# Hygiene-only public download surface for handoff artifacts. Do not extend.
"""Public (no-auth) download router for the 10 ClinEvidence handoff artifacts.

This router serves a tight allow-list of files from ``/app/memory/`` so the
operator can download the handoff bundle directly from the preview origin.

Scope is intentionally narrow:
- 10 files only, all timestamped ``20260608T214145Z``.
- Secrets scan returned 0 on all of them before this surface was created.
- Any unknown filename → 404. Any path-traversal attempt → 400.
- No auth dependency; no token required.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(prefix="/handoff", tags=["handoff"])

# Default keeps the Emergent container layout; override for off-platform runs.
_MEMORY_DIR = Path(os.environ.get("HANDOFF_MEMORY_DIR", "/app/memory"))
_TS = "20260608T214145Z"

# Closed-set allow-list. Adding a new entry here is the ONLY supported way
# to publish another file. Order is preserved for the HTML index.
_DOCX_MEDIA = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
ALLOWED_FILES: dict[str, dict[str, str]] = {
    f"ClinEvidence_Handoff_Bundle_{_TS}.zip": {
        "media_type": "application/zip",
        "label": "Handoff Bundle (ZIP — all 9 artifacts)",
    },
    f"ClinEvidence_Handoff_Bundle_{_TS}.pdf": {
        "media_type": "application/pdf",
        "label": "Handoff Bundle (PDF)",
    },
    f"ClinEvidence_Handoff_Bundle_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "Handoff Bundle (DOCX)",
    },
    f"ClinEvidence_Handoff_Bundle_{_TS}.md": {
        "media_type": "text/markdown; charset=utf-8",
        "label": "Handoff Bundle (Markdown)",
    },
    f"ClinEvidence_Current_Handoff_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "Master Handoff (DOCX)",
    },
    f"ClinEvidence_Current_Scope_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "Current Scope (DOCX)",
    },
    f"ClinEvidence_New_Emergent_Chat_Opening_Prompt_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "New Emergent Chat Opening Prompt (DOCX)",
    },
    f"ClinEvidence_Closeout_Index_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "Closeout Index (DOCX)",
    },
    f"ClinEvidence_Test_Snapshot_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "Test Snapshot (DOCX)",
    },
    f"phaseHandoff_current_state_for_new_emergent_chat_{_TS}.docx": {
        "media_type": _DOCX_MEDIA,
        "label": "Meta-Closeout (DOCX)",
    },
}


def _validate(filename: str) -> Path:
    """Closed-set validation → resolved absolute Path in /app/memory/.

    Order:
      1. Reject path-traversal markers (`..`, `/`, `\\`) → 400.
      2. Reject unknown filenames (not in ALLOWED_FILES) → 404.
      3. Confirm the resolved path is strictly inside /app/memory/ → 400.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    meta = ALLOWED_FILES.get(filename)
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown handoff artifact.")
    path = (_MEMORY_DIR / filename).resolve()
    try:
        path.relative_to(_MEMORY_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact missing on disk.")
    return path


@router.api_route("/list", methods=["GET", "HEAD"])
async def list_artifacts() -> dict:
    """Return the allow-listed artifacts with size + sha256-prefix metadata."""
    files: list[dict] = []
    for name in ALLOWED_FILES:
        path = _MEMORY_DIR / name
        if not path.is_file():
            files.append({"name": name, "size_bytes": None, "sha256_prefix8": None})
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256_prefix8": digest[:8],
            }
        )
    return {
        "files": files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.api_route("/download/{filename}", methods=["GET", "HEAD"])
async def download_artifact(filename: str) -> FileResponse:
    """Serve a single allow-listed artifact as a forced attachment."""
    path = _validate(filename)
    meta = ALLOWED_FILES[filename]
    return FileResponse(
        path=str(path),
        media_type=meta["media_type"],
        filename=filename,  # starlette sets Content-Disposition: attachment.
    )


@router.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Plain-HTML index listing the 10 handoff artifacts as download links."""
    items: list[str] = []
    for name, meta in ALLOWED_FILES.items():
        items.append(
            f'<li><a href="/api/handoff/download/{name}">{meta["label"]}</a>'
            f' &mdash; <code>{name}</code></li>'
        )
    body = (
        "<!doctype html>"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>ClinEvidence Search &mdash; Handoff Artifacts</title>"
        "</head><body>"
        "<h1>ClinEvidence Search &mdash; Handoff Artifacts</h1>"
        f"<p>Timestamp: <code>{_TS}</code>. "
        "Click any link to download. These files contain no secrets.</p>"
        "<ul>" + "".join(items) + "</ul>"
        "<p><a href=\"/api/handoff/list\">JSON listing (with sizes + sha256 prefixes)</a></p>"
        "</body></html>"
    )
    return HTMLResponse(content=body, status_code=200)
