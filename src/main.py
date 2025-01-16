from fastapi import FastAPI, Body
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from iFinD import IFinD

app = FastAPI(
    title="vFinD backend",
    description="vFinD 后端服务",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    # 允许的源列表
    allow_origins=["*"],  # 允许所有源
    # 允许的HTTP方法
    allow_methods=["*"],  # 允许所有方法
    # 允许的HTTP头
    allow_headers=["*"],  # 允许所有头
    # 是否允许发送cookie
    allow_credentials=True
)


@app.get("/")
async def root():
    return {"message": "欢迎使用 vFinD"}


@app.post("/login")
async def login(username: str = Body(...), password: str = Body(...)):
    print(username, password)
    iFind = IFinD(username, password)
    cookie = iFind.cookie
    return {"cookie": cookie}


# 启动服务器
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
