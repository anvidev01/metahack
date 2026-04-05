from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models import Action, Observation, Reward
from app.env import IndiaServiceEnv
from app.tasks import TASKS_CONFIG

app = FastAPI(title="IndiaServiceEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance. In a real multi-tenant deployment we'd use sessions or stateful managers.
# For OpenEnv HF space standard, usually a single instance is tested continuously per evaluate call,
# or we just re-instantiate it on reset. But to support /state and /step we keep a global ref.
# To be robust, let's keep one instance per task_id or just one global active instance.
active_env = None

from typing import Optional

class ResetRequest(BaseModel):
    task_id: Optional[str] = "classify_and_route"

@app.post("/reset", response_model=Observation)
def reset_env(req: Optional[ResetRequest] = None):
    global active_env
    if req is None:
        req = ResetRequest()
    if req.task_id not in TASKS_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid task_id")
    
    active_env = IndiaServiceEnv(task_id=req.task_id)
    obs = active_env.reset()
    return obs

@app.post("/step")
def step_env(action: Action):
    global active_env
    if active_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs, reward, done, info = active_env.step(action)
    
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": reward.model_dump() if hasattr(reward, "model_dump") else reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    global active_env
    if active_env is None:
        return {}
    return active_env.state()

@app.get("/tasks")
def get_tasks():
    return list(TASKS_CONFIG.keys())

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "IndiaServiceEnv API is running. Use /reset, /step, /state to interact."}
