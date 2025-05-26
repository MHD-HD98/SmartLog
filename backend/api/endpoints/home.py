# app/api/endpoints/home.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def get_home():
    with open("app/static/index.html", "r") as file:
        return HTMLResponse(content=file.read())
