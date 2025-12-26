""" 
Research Project Intake Server
==============================
Local server for the research project intake form.
Creates project folders and saves initial project data.

Run: python scripts/research_intake_server.py
Then open: http://localhost:8080

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime
from email.parser import BytesParser
from email.policy import HTTP
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import uuid


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import FILENAMES, INTAKE_SERVER  # noqa: E402
from src.utils.zip_safety import extract_zip_bytes_safely  # noqa: E402

USER_INPUT_DIR = str(ROOT_DIR / "user-input")
STATIC_DIR = str(ROOT_DIR)

# Use centralized config for safety limits
PORT = INTAKE_SERVER.PORT
MAX_UPLOAD_MB = INTAKE_SERVER.MAX_UPLOAD_MB
MAX_ZIP_FILES = INTAKE_SERVER.MAX_ZIP_FILES
MAX_ZIP_TOTAL_MB = INTAKE_SERVER.MAX_ZIP_TOTAL_MB


class ResearchIntakeHandler(SimpleHTTPRequestHandler):
    """Handle research project intake form submissions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/research_intake_form.html"
        return super().do_GET()

    def do_POST(self):
        if self.path == "/submit":
            self.handle_submission()
        else:
            self.send_error(404, "Not Found")

    def handle_submission(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_error(400, "Expected multipart/form-data")
                return

            content_length = int(self.headers.get("Content-Length", 0))
            max_bytes = MAX_UPLOAD_MB * 1024 * 1024
            if content_length <= 0:
                self.send_error(400, "Missing Content-Length")
                return
            if content_length > max_bytes:
                self.send_error(413, f"Upload too large (max {MAX_UPLOAD_MB} MB)")
                return

            body = self.rfile.read(content_length)

            form_data, files = self.parse_multipart(content_type, body)

            project_data = {
                "id": str(uuid.uuid4())[:8],
                "created_at": datetime.now().isoformat(),
                "title": form_data.get("title", ""),
                "research_question": form_data.get("research_question", ""),
                "has_hypothesis": form_data.get("has_hypothesis", "") == "yes",
                "hypothesis": form_data.get("hypothesis", ""),
                "target_journal": form_data.get("target_journal", ""),
                "paper_type": form_data.get("paper_type", ""),
                "research_type": form_data.get("research_type", ""),
                "has_data": form_data.get("has_data", "") == "yes",
            }

            project_folder = self.create_project_folder(project_data)
            if files:
                self.handle_file_uploads(project_folder, files)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "project_folder": project_folder}).encode())
        except Exception as e:
            self.send_error(500, f"Server error: {e}")

    def parse_multipart(self, content_type: str, body: bytes):
        header_data = f"Content-Type: {content_type}\r\n\r\n".encode()
        msg = BytesParser(policy=HTTP).parsebytes(header_data + body)

        form_data = {}
        files = {}

        for part in msg.iter_parts():
            disp = part.get("Content-Disposition", "")
            name_match = re.search(r'name="([^"]+)"', disp)
            if not name_match:
                continue
            field_name = name_match.group(1)

            filename_match = re.search(r'filename="([^"]*)"', disp)
            if filename_match and filename_match.group(1):
                files[field_name] = {
                    "filename": filename_match.group(1),
                    "content": part.get_payload(decode=True) or b"",
                }
            else:
                payload = part.get_payload(decode=True) or b""
                form_data[field_name] = payload.decode(errors="replace")

        return form_data, files

    def create_project_folder(self, project_data: dict) -> str:
        project_id = project_data.get("id", "unknown")
        title = project_data.get("title", "project")
        slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", title.strip().lower()).strip("-")
        slug = slug[:60] if slug else "project"

        folder_name = f"{project_id}_{slug}"
        project_path = Path(USER_INPUT_DIR) / folder_name
        project_path.mkdir(parents=True, exist_ok=True)

        (project_path / "data" / "raw data").mkdir(parents=True, exist_ok=True)
        (project_path / "literature").mkdir(parents=True, exist_ok=True)
        (project_path / "drafts").mkdir(parents=True, exist_ok=True)
        (project_path / "paper").mkdir(parents=True, exist_ok=True)

        with open(project_path / "project.json", "w") as f:
            json.dump(project_data, f, indent=2)

        with open(project_path / "README.md", "w") as f:
            f.write(f"# {project_data.get('title', 'Research Project')}\n\n")
            f.write(f"Project ID: {project_id}\n")
            f.write(f"Created: {project_data.get('created_at', '')}\n\n")

        return str(project_path)

    def handle_file_uploads(self, project_folder: str, files: dict) -> None:
        project_path = Path(project_folder)
        for _, file_info in files.items():
            filename = file_info.get("filename", "")
            content = file_info.get("content", b"")
            if not filename or not content:
                continue

            if filename.lower().endswith(".zip"):
                self.extract_zip_safely(project_path, content)
            else:
                safe_name = re.sub(r"[^a-zA-Z0-9._\-]+", "_", filename)
                safe_name = safe_name[: FILENAMES.MAX_LENGTH] or "upload"
                dest = project_path / "data" / "raw data" / safe_name
                dest.write_bytes(content)

    def extract_zip_safely(self, project_path: Path, content: bytes) -> None:
        tmp_dir = project_path / "_tmp_extract"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Avoid zipfile.extract; it is historically easy to misuse and can enable
            # path traversal when archives contain ".." or absolute paths.
            result = extract_zip_bytes_safely(
                content=content,
                dest_dir=tmp_dir,
                max_files=int(MAX_ZIP_FILES),
                max_total_uncompressed_bytes=int(MAX_ZIP_TOTAL_MB) * 1024 * 1024,
                max_filename_length=int(FILENAMES.MAX_LENGTH),
            )
            if result.truncated or result.skipped_entries:
                self.log_message(
                    "ZIP extraction incomplete (truncated=%s, skipped_entries=%s) for project %s",
                    bool(result.truncated),
                    int(result.skipped_entries),
                    str(project_path),
                )
            raw_data_dir = project_path / "data" / "raw data"
            for item in tmp_dir.rglob("*"):
                if item.is_file():
                    safe_name = re.sub(r"[^a-zA-Z0-9._\-]+", "_", item.name)
                    safe_name = safe_name[: FILENAMES.MAX_LENGTH] or "upload"
                    shutil.copy2(item, raw_data_dir / safe_name)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)


def main() -> None:
    server = HTTPServer(("", PORT), ResearchIntakeHandler)
    print(f"Research intake server running on http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
