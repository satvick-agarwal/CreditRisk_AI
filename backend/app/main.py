from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.core.database import init_db
from backend.app.services.ml_service import ml_service
from backend.app.api.endpoints import router as api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize DB
    await init_db()
    
    # Load ML artifacts
    try:
        ml_service.load_artifacts()
    except Exception as e:
        print(f"WARNING: Could not load ML artifacts: {e}")
        
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": ml_service.is_loaded,
        "version": settings.APP_VERSION
    }
