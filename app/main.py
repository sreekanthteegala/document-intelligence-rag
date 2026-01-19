from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from app.api.routes import router

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

app.include_router(router)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
