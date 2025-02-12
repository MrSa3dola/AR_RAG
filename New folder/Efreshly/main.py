from fastapi import FastAPI
from finalp_llm import app as furniture_router

app = FastAPI(title="Furniture Recommendation API")

# Include the router with a prefix (for example, /api)
app.include_router(furniture_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
