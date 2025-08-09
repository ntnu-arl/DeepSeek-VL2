from typing import Dict

job_store: Dict[str, Dict] = {}


def create_job(job_id: str):
    job_store[job_id] = {"status": "pending", "result": None}


def update_job(job_id: str, result):
    job_store[job_id]["status"] = "completed"
    job_store[job_id]["result"] = result


def fail_job(job_id: str, error: str):
    job_store[job_id]["status"] = "failed"
    job_store[job_id]["error"] = error


def get_job(job_id: str):
    return job_store.get(job_id)
