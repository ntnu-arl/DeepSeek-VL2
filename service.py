import torch
from typing import List
from model import DeepSeekVL2
from config import DeepSeekVL2Config, DEVICE
from jobs import update_job, fail_job
import time

model = DeepSeekVL2(DeepSeekVL2Config())
model.to(DEVICE)


def run_inference(job_id: str, prompts: List[str], features: List[List[List[float]]], deterministic: bool):
    try:
        start_time = time.time()
        features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        outputs = []
        for i in range(len(prompts)):
            result = model.generate_caption(features_tensor[i][None, ...], [prompts[i]], deterministic=deterministic)
            outputs.append(result[0])
        print(
            f"Inference time for job {job_id}: {time.time() - start_time:.2f} seconds"
        )
        print(f"Results for job {job_id}: {outputs}")
        update_job(job_id, outputs)
    except Exception as e:
        fail_job(job_id, str(e))
