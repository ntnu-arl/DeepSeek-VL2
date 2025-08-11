from typing import List
from pydantic import BaseModel


class VLMInput(BaseModel):
    deterministic: bool = True
    prompts: List[str]
    features: List[List[List[float]]]


class JobResponse(BaseModel):
    job_id: str
