from typing import List
from pydantic import BaseModel


class VLMInput(BaseModel):
    prompts: List[str]
    features: List[List[List[float]]]


class JobResponse(BaseModel):
    job_id: str
