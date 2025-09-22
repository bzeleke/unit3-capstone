from flask import Flask, request, jsonify
import os, math
from typing import List
from sentence_transformers import SentenceTransformer
app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
def _load():
    global _model
    if _model is None:
        src = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME
        _model = SentenceTransformer(src)
    return _model
def _l2(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]
@app.get("/healthz")
def healthz(): return jsonify({"ok": True})
@app.post("/embed")
def embed():
    body = request.get_json(force=True, silent=True) or {}
    texts = body.get("texts"); normalize = body.get("normalize", True)
    if not isinstance(texts, list) or not all(isinstance(t,str) for t in texts):
        return jsonify({"error":"Expected {'texts':[str,...], 'normalize': bool}"}), 400
    m = _load()
    vecs = m.encode(texts, convert_to_numpy=True).tolist()
    if normalize: vecs = [_l2(v) for v in vecs]
    return jsonify({"vectors": vecs, "dimension": len(vecs[0]) if vecs else 384})
if __name__ == "__main__": app.run(host="0.0.0.0", port=8080)
