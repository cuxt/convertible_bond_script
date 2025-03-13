from fastapi import FastAPI, Body
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from iFinD import IFinD

app = FastAPI(
    title="vFinD backend",
    description="vFinD API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


@app.get("/")
async def root():
    return {"message": "欢迎使用 vFinD"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/login")
async def login(username: str = Body(...), password: str = Body(...)):
    print(username, password)
    iFind = IFinD(username, password)
    cookie = iFind.cookie
    return {"cookie": cookie}


# 启动服务器
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
