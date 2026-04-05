from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dashscope import MultiModalEmbedding
import dashscope
import uvicorn
import os

from starlette.responses import JSONResponse

app = FastAPI()

class EmbeddingRequest(BaseModel):
    img_base64: str
    text: str


@app.post("/get_vector")
async def get_vector(request: EmbeddingRequest):
    try:
        # 1. 构造输入
        input_data = [
            {'image': request.img_base64},
            {'text': request.text}
        ]

        # 2. 调用阿里 DashScope SDK
        resp = MultiModalEmbedding.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen3-vl-embedding",
            input=input_data,
            dimension = 1024,
            enable_fusion=True,
        )

        print("--- DashScope API Response Start ---")
        print(resp)
        print("--- DashScope API Response End ---")

        if resp.status_code == 200:
            output = resp.output

            if 'embeddings' in output:
                vector = output['embeddings'][0]['embedding']
                return {"embedding": vector}

            elif 'embedding' in output:
                return {"embedding": output['embedding']}

            else:
                return JSONResponse(status_code=500, content={"detail": f"未在返回中找到向量字段: {str(output)}"})
        else:
            return JSONResponse(status_code=resp.status_code, content={"detail": resp.message})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)