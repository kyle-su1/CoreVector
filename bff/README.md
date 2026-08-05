# CoreVector Visualization Stack

An interactive web frontend for exploring the vector index: a 3D embedding
cloud (PCA 128D → 3D) plus a k-NN search playground.

```
Browser (React + deck.gl)  ──/api──▶  FastAPI BFF (:8000)  ──gRPC──▶  corevector_server (:50051)
```

The browser cannot speak native gRPC, so the FastAPI BFF bridges HTTP/JSON to
gRPC and does the dimensionality reduction (numpy SVD PCA), sending the browser
only 3 coordinates per point.

## Running (three terminals, from the project root)

```bash
# 1. C++ gRPC backend
./build/corevector_server

# 2. BFF (needs the venv deps: pip install -r requirements.txt)
venv/bin/uvicorn bff.app:app --port 8000 --reload

# 3. Frontend dev server
cd frontend && npm install && npm run dev
```

Then open the Vite URL (http://localhost:5173, or 5174 if 5173 is taken).
Click **Seed demo data** to populate the index, then search by point id.

Seed from the CLI instead of the button:

```bash
venv/bin/python bff/seed.py 300
```

## Endpoints

| Method | Path           | Purpose                                                    |
| ------ | -------------- | ---------------------------------------------------------- |
| GET    | `/api/vectors` | All vectors projected to 3D `{id, x, y, z, payload}`       |
| POST   | `/api/search`  | k-NN by `id` or `query`; returns neighbors + projected query |
| POST   | `/api/insert`  | Insert vectors (invalidates the cached PCA basis)          |
| GET    | `/api/health`  | Liveness + configured gRPC target                          |

The PCA basis is fit once from all vectors and cached, so the query point and
the cloud share the same 3D coordinate space. Inserting new vectors invalidates
the cache; it refits lazily on the next `/api/vectors` call.
