from fastapi import FastAPI
from env.grid_env import GridEnv
from env.models import Action

app = FastAPI()
env = GridEnv()

@app.get("/")
def home():
    return {"status": "GridMind running"}
@app.post("/reset")
async def reset():
    return (await env.reset()).dict()


@app.post("/step")
async def step(action: dict):
    return (await env.step(Action(**action))).dict()


@app.get("/state")
def state():
    return env.state().dict()
def main():
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()