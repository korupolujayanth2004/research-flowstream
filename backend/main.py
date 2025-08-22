from fastapi import FastAPI
from backend.api.routes import router  # absolute import
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Multi-Agent Research Assistant Backend")

# Register routes
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Backend is running. Check /docs for API details."}
