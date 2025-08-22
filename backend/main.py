from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Updated import - since 'api' folder will be directly under /app in the container
from api.routes import router

app = FastAPI(title="Multi-Agent Research Assistant Backend")

# Register routes
app.include_router(router)

# CORS middleware
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
