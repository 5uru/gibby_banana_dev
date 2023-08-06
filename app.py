import torch
from potassium import Potassium, Request, Response
from transformers import pipeline
app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model_name = "daryl149/Llama-2-7b-chat-hf"
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("text-generation", model=model_name, device=device)

    context = {
        "model": model
    }

    return context


# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    do_sample = request.json.get("do_sample", True)
    max_new_tokens = request.json.get("max_new_tokens", 256)
    top_p = request.json.get("top_p", 0.92)
    top_k = request.json.get("top_k", 0)

    model = context.get("model")
    outputs = model(prompt, do_sample=do_sample, max_new_tokens=int(max_new_tokens), top_k=float(top_k), top_p=float(top_p))

    return Response(
        json = {"outputs": outputs[0]},
        status=200
    )


if __name__ == "__main__":
    app.serve()
