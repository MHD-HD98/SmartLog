# main.py

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import face, stream, websocket
from core.config import settings

app = FastAPI(title="Face Recognition API")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routers
app.include_router(face.router)
app.include_router(stream.router)
app.include_router(websocket.router)


# Home route (static HTML page)
@app.get("/", include_in_schema=False)
def get_home():
    with open("static/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
