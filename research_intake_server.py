"""
Research Project Intake Server
==============================
Local server for the research project intake form.
Creates project folders and saves initial project data.

Run: python research_intake_server.py
Then open: http://localhost:8080

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import sys
import json
import uuid
import shutil
import zipfile
import threading
import re
import io
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
from email.parser import BytesParser
from email.policy import HTTP

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Configuration
PORT = 8080
USER_INPUT_DIR = os.path.join(os.path.dirname(__file__), "user-input")
STATIC_DIR = os.path.dirname(__file__)


class ResearchIntakeHandler(SimpleHTTPRequestHandler):
    """Handle research project intake form submissions."""
    
    def do_GET(self):
        """Serve the intake form or static files."""
        if self.path == "/" or self.path == "/index.html":
            self.path = "/research_intake_form.html"
        return super().do_GET()
    
    def do_POST(self):
        """Handle form submission."""
        if self.path == "/submit":
            self.handle_submission()
        else:
            self.send_error(404, "Not Found")
    
    def handle_submission(self):
        """Process the intake form submission."""
        try:
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            
            if "multipart/form-data" in content_type:
                # Parse multipart data without cgi module (removed in Python 3.13)
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                
                # Parse the multipart data
                form_data, files = self.parse_multipart(content_type, body)
                
                # Extract form fields
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
                    "data_description": form_data.get("data_description", ""),
                    "data_sources": form_data.get("data_sources", ""),
                    "key_variables": form_data.get("key_variables", ""),
                    "methodology": form_data.get("methodology", ""),
                    "related_literature": form_data.get("related_literature", ""),
                    "expected_contribution": form_data.get("expected_contribution", ""),
                    "constraints": form_data.get("constraints", ""),
                    "deadline": form_data.get("deadline", ""),
                    "additional_notes": form_data.get("additional_notes", ""),
                    "uploaded_files": [],
                }
                
                # Create project folder
                project_slug = self.create_slug(project_data["title"])
                project_folder = os.path.join(
                    USER_INPUT_DIR, 
                    f"{project_data['id']}_{project_slug}"
                )
                os.makedirs(project_folder, exist_ok=True)
                os.makedirs(os.path.join(project_folder, "data"), exist_ok=True)
                os.makedirs(os.path.join(project_folder, "literature"), exist_ok=True)
                os.makedirs(os.path.join(project_folder, "drafts"), exist_ok=True)
                
                # Handle file uploads (ZIP only, auto-extract)
                data_dir = os.path.join(project_folder, "data")
                MAX_EXTRACT_SIZE = 500 * 1024 * 1024  # 500MB max extracted size
                for filename, file_content in files.items():
                    if filename:
                        # Sanitize filename
                        safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', os.path.basename(filename))
                        if not safe_filename or safe_filename.startswith('.'):
                            continue
                        
                        # Only process ZIP files
                        if not safe_filename.lower().endswith('.zip'):
                            continue
                        
                        # Save the ZIP temporarily
                        temp_zip = os.path.join(data_dir, safe_filename)
                        with open(temp_zip, "wb") as f:
                            f.write(file_content)
                        
                        # Extract ZIP contents with security checks
                        try:
                            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                                # Check for ZIP bomb (excessive compression ratio)
                                total_size = sum(info.file_size for info in zip_ref.infolist())
                                if total_size > MAX_EXTRACT_SIZE:
                                    raise ValueError(f"ZIP extraction size exceeds limit: {total_size} bytes")
                                
                                # Check for path traversal
                                for member in zip_ref.namelist():
                                    member_path = os.path.normpath(member)
                                    if member_path.startswith('..') or os.path.isabs(member_path):
                                        raise ValueError(f"Invalid path in ZIP: {member}")
                                
                                # Extract to a subfolder named after the zip
                                extract_name = safe_filename[:-4]  # Remove .zip
                                extract_dir = os.path.join(data_dir, extract_name)
                                zip_ref.extractall(extract_dir)
                                
                                # List extracted files
                                extracted_files = []
                                for root, dirs, fnames in os.walk(extract_dir):
                                    for fname in fnames:
                                        rel_path = os.path.relpath(
                                            os.path.join(root, fname), data_dir
                                        )
                                        extracted_files.append(rel_path)
                                
                                project_data["uploaded_files"].append({
                                    "archive": filename,
                                    "extracted_to": extract_name,
                                    "files": extracted_files
                                })
                            # Remove the ZIP after extraction
                            os.remove(temp_zip)
                        except zipfile.BadZipFile:
                            # Keep the file if it's not a valid ZIP
                            project_data["uploaded_files"].append({
                                "archive": filename,
                                "error": "Invalid ZIP file"
                            })
                
                # Save project metadata
                metadata_path = os.path.join(project_folder, "project.json")
                with open(metadata_path, "w") as f:
                    json.dump(project_data, f, indent=2)
                
                # Create README for the project
                readme_content = self.generate_readme(project_data)
                readme_path = os.path.join(project_folder, "README.md")
                with open(readme_path, "w") as f:
                    f.write(readme_content)
                
                # Send success response immediately
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {
                    "success": True,
                    "project_id": project_data["id"],
                    "project_folder": project_folder,
                    "message": f"Project '{project_data['title']}' created. Starting analysis...",
                    "workflow_status": "starting"
                }
                self.wfile.write(json.dumps(response).encode())
                
                print(f"\n[OK] Created project: {project_folder}")
                
                # Start the agent workflow in a background thread
                def run_workflow_background(folder):
                    try:
                        print(f"\n[>>] Starting agent workflow for: {folder}")
                        from src.agents.workflow import run_workflow_sync
                        result = run_workflow_sync(folder)
                        if result.success:
                            print(f"\n[OK] Workflow completed: {result.overview_path}")
                        else:
                            print(f"\n[!!] Workflow completed with errors: {result.errors}")
                    except Exception as e:
                        print(f"\n[!!] Workflow error: {e}")
                
                thread = threading.Thread(
                    target=run_workflow_background,
                    args=(project_folder,),
                    daemon=True
                )
                thread.start()
                
            else:
                raise ValueError("Invalid content type")
                
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"success": False, "error": str(e)}
            self.wfile.write(json.dumps(response).encode())
            print(f"\n[!!] Error: {e}")
    
    def parse_multipart(self, content_type, body):
        """
        Parse multipart form data without cgi module (removed in Python 3.13).
        
        Returns:
            tuple: (form_data dict, files dict)
        """
        form_data = {}
        files = {}
        
        # Extract boundary from content-type
        boundary_match = re.search(r'boundary=([^;\s]+)', content_type)
        if not boundary_match:
            return form_data, files
        
        boundary = boundary_match.group(1).encode()
        if boundary.startswith(b'"') and boundary.endswith(b'"'):
            boundary = boundary[1:-1]
        
        # Split body by boundary
        parts = body.split(b'--' + boundary)
        
        for part in parts:
            if not part or part == b'--\r\n' or part == b'--':
                continue
            
            # Split headers from content
            if b'\r\n\r\n' in part:
                headers_raw, content = part.split(b'\r\n\r\n', 1)
            else:
                continue
            
            # Remove trailing boundary markers
            if content.endswith(b'\r\n'):
                content = content[:-2]
            
            # Parse headers
            headers_str = headers_raw.decode('utf-8', errors='replace')
            
            # Extract field name
            name_match = re.search(r'name="([^"]+)"', headers_str)
            if not name_match:
                continue
            field_name = name_match.group(1)
            
            # Check if it's a file upload
            filename_match = re.search(r'filename="([^"]*)"', headers_str)
            if filename_match:
                filename = filename_match.group(1)
                if filename:  # Only process if filename is not empty
                    files[filename] = content
            else:
                # Regular form field
                try:
                    form_data[field_name] = content.decode('utf-8')
                except UnicodeDecodeError:
                    form_data[field_name] = content.decode('latin-1')
        
        return form_data, files
    
    def create_slug(self, title):
        """Create a URL-friendly slug from title."""
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug[:50].strip('-')
    
    def generate_readme(self, data):
        """Generate a README.md for the project."""
        readme = f"""# {data['title']}

**Project ID:** {data['id']}  
**Created:** {data['created_at'][:10]}  
**Target Journal:** {data['target_journal']}  
**Paper Type:** {data['paper_type']}

---

## Research Question

{data['research_question']}

"""
        if data['has_hypothesis']:
            readme += f"""## Hypothesis

{data['hypothesis']}

"""
        
        readme += f"""## Research Type

{data['research_type']}

## Methodology

{data['methodology'] or 'To be determined'}

## Key Variables

{data['key_variables'] or 'To be determined'}

## Data

**Has Data:** {'Yes' if data['has_data'] else 'No'}

{data['data_description'] or ''}

**Data Sources:** {data['data_sources'] or 'To be determined'}

"""
        
        if data['uploaded_files']:
            readme += "**Uploaded Data:**\n"
            for f in data['uploaded_files']:
                if isinstance(f, dict):
                    if 'error' in f:
                        readme += f"- `{f['archive']}` (error: {f['error']})\n"
                    else:
                        readme += f"- `{f['archive']}` extracted to `data/{f['extracted_to']}/`\n"
                        for extracted in f.get('files', [])[:10]:  # Show first 10 files
                            readme += f"  - `{extracted}`\n"
                        if len(f.get('files', [])) > 10:
                            readme += f"  - ... and {len(f['files']) - 10} more files\n"
                else:
                    readme += f"- `data/{f}`\n"
            readme += "\n"
        
        readme += f"""## Related Literature

{data['related_literature'] or 'To be reviewed'}

## Expected Contribution

{data['expected_contribution'] or 'To be determined'}

## Constraints and Requirements

{data['constraints'] or 'None specified'}

## Deadline

{data['deadline'] or 'Not specified'}

## Additional Notes

{data['additional_notes'] or 'None'}

---

## Project Structure

```
{data['id']}_{self.create_slug(data['title'])}/
├── README.md          # This file
├── project.json       # Project metadata
├── data/              # Raw data files
├── literature/        # Reference papers
└── drafts/            # Paper drafts
```

---

*Project created via Research Intake System*
"""
        return readme


def main():
    """Start the research intake server."""
    # Ensure user-input directory exists
    os.makedirs(USER_INPUT_DIR, exist_ok=True)
    
    # Change to static directory
    os.chdir(STATIC_DIR)
    
    # Start server
    server = HTTPServer(("localhost", PORT), ResearchIntakeHandler)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║          Research Project Intake Server                    ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Server running at: http://localhost:{PORT}                 ║
║                                                            ║
║  Open this URL in your browser to start a new project.     ║
║  Press Ctrl+C to stop the server.                          ║
║                                                            ║
║  Projects will be saved to: {USER_INPUT_DIR[:30]}...       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
