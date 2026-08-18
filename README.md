# Resume Score Point Parser

An AI-powered ATS (Applicant Tracking System) parser built with Python, Groq LLM API, and UV. It extracts structured information from job descriptions and candidate resumes (`.pdf` and `.docx`), evaluates candidate fit, and ranks the top candidates dynamically.

## Features

- **Dynamic File Reading:** Automatically extracts job descriptions from plain text (`.txt`) and parses resumes in `.pdf` and `.docx` formats.
- **Structured LLM Extraction:** Uses **Pydantic** models with **Groq LLM** (`openai/gpt-oss-20b`) to extract structured candidate details and job requirements in strictly formatted JSON.
- **Automated Scoring:** Evaluates candidate resumes against target job roles, generating a match score (0–100) along with key insights (matching skills, missing skills, experience evaluation).
- **Fast & Modern Package Management:** Managed completely with **UV** for fast, reliable dependency resolution.

---

## Project Structure

```text
RoleFitAi/
├── .env                  # API keys and environment variables (ignored by Git)
├── .gitignore            # Git ignore rules
├── job_description.txt   # Target job description input file
├── main.py               # Core application logic
├── pyproject.toml        # UV project definition & dependencies
├── README.md             # Project documentation
├── uv.lock               # Locked dependency versions
└── resumes/              # Folder containing candidate resumes (.pdf / .docx)
    ├── candidate1.pdf
    └── candidate2.docx