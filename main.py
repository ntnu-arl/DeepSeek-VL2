from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from schemas import VLMInput, JobResponse
from auth import verify_api_key
from service import run_inference
from jobs import create_job, get_job
import uuid

app = FastAPI()


@app.post("/generate/", response_model=JobResponse)
async def generate(
    input_data: VLMInput,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    job_id = str(uuid.uuid4())
    create_job(job_id)
    background_tasks.add_task(
        run_inference, job_id, input_data.prompts, input_data.features, input_data.deterministic
    )
    return JobResponse(job_id=job_id)


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job
