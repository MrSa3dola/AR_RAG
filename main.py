import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from finalp_llm import app as furniture_router
from routes.views import router as views_router  # Import the router from views.py

dotenv.load_dotenv()

app = FastAPI()

# Include routers
app.include_router(furniture_router)
app.include_router(views_router)

# Mount static directory
app.mount("/static", StaticFiles(directory="assets"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


print("test")