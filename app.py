import torch
from potassium import Potassium, Request, Response
from transformers import pipeline
app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model_name = "mosaicml/mpt-7b-chat"
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

    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]},
        status=200
    )


if __name__ == "__main__":
    app.serve()
