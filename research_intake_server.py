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
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import cgi

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
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    }
                )
                
                # Extract form fields
                project_data = {
                    "id": str(uuid.uuid4())[:8],
                    "created_at": datetime.now().isoformat(),
                    "title": self.get_field(form, "title"),
                    "research_question": self.get_field(form, "research_question"),
                    "has_hypothesis": self.get_field(form, "has_hypothesis") == "yes",
                    "hypothesis": self.get_field(form, "hypothesis"),
                    "target_journal": self.get_field(form, "target_journal"),
                    "paper_type": self.get_field(form, "paper_type"),
                    "research_type": self.get_field(form, "research_type"),
                    "has_data": self.get_field(form, "has_data") == "yes",
                    "data_description": self.get_field(form, "data_description"),
                    "data_sources": self.get_field(form, "data_sources"),
                    "key_variables": self.get_field(form, "key_variables"),
                    "methodology": self.get_field(form, "methodology"),
                    "related_literature": self.get_field(form, "related_literature"),
                    "expected_contribution": self.get_field(form, "expected_contribution"),
                    "constraints": self.get_field(form, "constraints"),
                    "deadline": self.get_field(form, "deadline"),
                    "additional_notes": self.get_field(form, "additional_notes"),
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
                if "data_files" in form:
                    files = form["data_files"]
                    if not isinstance(files, list):
                        files = [files]
                    
                    data_dir = os.path.join(project_folder, "data")
                    for file_item in files:
                        if file_item.filename:
                            filename = os.path.basename(file_item.filename)
                            # Only process ZIP files
                            if not filename.lower().endswith('.zip'):
                                continue
                            
                            # Save the ZIP temporarily
                            temp_zip = os.path.join(data_dir, filename)
                            with open(temp_zip, "wb") as f:
                                f.write(file_item.file.read())
                            
                            # Extract ZIP contents
                            try:
                                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                                    # Extract to a subfolder named after the zip
                                    extract_name = filename[:-4]  # Remove .zip
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
    
    def get_field(self, form, name):
        """Safely get a field value from the form."""
        if name in form:
            field = form[name]
            if isinstance(field, list):
                return field[0].value if field else ""
            return field.value if field.value else ""
        return ""
    
    def create_slug(self, title):
        """Create a URL-friendly slug from title."""
        import re
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
