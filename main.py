import json
import os
import time
from pathlib import Path
from docx import Document
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

client = Groq(api_key=API_KEY)
MODEL_NAME = "openai/gpt-oss-20b"

class JobD(BaseModel):
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

class MatchResult(BaseModel):
    score: float
    details: dict

def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def read_docx(file_path: Path) -> str:
    doc = Document(file_path)
    text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text.append(cell.text.strip())
    return "\n".join(text)

def extract_resume_text(file_path: Path) -> str | None:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(file_path)
    elif suffix == ".docx":
        return read_docx(file_path)
    return None

def parse_job_description(jd_text: str) -> JobD:
    schema = JobD.model_json_schema()
    system_prompt = (
        "You are an expert HR assistant. Analyze job descriptions and extract structured details. "
        f"Return ONLY valid JSON strictly adhering to this schema:\n{schema}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": jd_text},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return JobD(**data)

def parse_resume(resume_text: str) -> Resume:
    schema = Resume.model_json_schema()
    system_prompt = (
        "You are an expert ATS parser. Extract information based on semantic meaning, "
        "not just exact heading names. Include internships in experience. "
        f"Return ONLY valid JSON matching this schema:\n{schema}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": resume_text},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return Resume(**data)

def evaluate_candidate(job: JobD, resume: Resume) -> MatchResult:
    prompt = f"""You are an HR recruiter evaluating a candidate. Compare the resume against the job requirements.

JOB REQUIREMENTS:
{job.model_dump_json(indent=2)}

CANDIDATE RESUME:
{resume.model_dump_json(indent=2)}

Return ONLY a JSON object with this shape:
{{
    "score": 85,
    "details": {{
        "candidate_name": "John Doe",
        "matching_skills": ["Python", "AWS"],
        "missing_skills": ["Java"],
        "experience_requirement_met": true,
        "verdict": "Short summary of evaluation"
    }}
}}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return MatchResult(**data)

def main():
    jd_path = Path("job_description.txt")
    resume_folder = Path("resumes")

    if not jd_path.exists():
        raise FileNotFoundError(f"Job description file '{jd_path}' not found.")

    if not resume_folder.exists():
        raise FileNotFoundError(f"Directory '{resume_folder}' not found.")

    print("Parsing Job Description...")
    job_description_text = jd_path.read_text(encoding="utf-8")
    parsed_job = parse_job_description(job_description_text)

    results = []
    for file_path in resume_folder.iterdir():
        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        print(f"Parsing resume: {file_path.name}")

        try:
            resume_text = extract_resume_text(file_path)
            if not resume_text:
                continue

            parsed_resume = parse_resume(resume_text)
            time.sleep(2)

            evaluation = evaluate_candidate(parsed_job, parsed_resume)
            time.sleep(2)

            results.append({
                "name": parsed_resume.name or file_path.stem,
                "score": evaluation.score,
                "details": evaluation.details,
            })
            print(f"Finished {file_path.name} - Score: {evaluation.score}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n--- TOP CANDIDATES ---")
    for candidate in results[:2]:
        print(f"{candidate['name']} - Score: {candidate['score']}")

    print("\n--- BOTTOM CANDIDATES ---")
    for candidate in results[-2:]:
        print(f"{candidate['name']} - Score: {candidate['score']}")

if __name__ == "__main__":
    main()