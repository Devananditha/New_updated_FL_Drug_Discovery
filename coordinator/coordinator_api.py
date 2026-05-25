"""Central coordinator for federated drug-target graph retrieval."""

import requests
import uvicorn
from fastapi import FastAPI

app = FastAPI()

CLIENT_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]


@app.get("/global_retrieve")
def global_retrieve(drug_id: str):
    """Query all lab clients and aggregate their raw responses."""
    raw_responses = []

    for url in CLIENT_URLS:
        target_url = f"{url}/retrieve?drug_id={drug_id}"
        response = requests.get(target_url)
        raw_responses.append(response.json())

    return {"query": drug_id, "raw_responses": raw_responses}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
