# User Input Directory

This directory stores research project intakes created via the Research Intake Form.

## Usage

1. Start the intake server:
   ```bash
   python research_intake_server.py
   ```

2. Open http://localhost:8080 in your browser

3. Fill out the research project intake form

4. Submit to create a new project folder here

## Project Structure

Each project folder contains:

```
{project_id}_{project-slug}/
├── README.md          # Project overview
├── project.json       # Full project metadata
├── data/              # Uploaded and raw data files
├── literature/        # Reference papers and sources
└── drafts/            # Paper drafts and revisions
```

## Project Metadata

The `project.json` file contains:
- Project ID and creation date
- Research question and hypothesis
- Target journal and paper type
- Data sources and methodology
- Related literature
- Expected contribution
- Timeline and constraints

---

*Created by Gia Tenica Research Intake System*
