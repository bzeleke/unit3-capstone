import os
import json
import boto3
import requests

REGION      = os.getenv("REGION", "us-east-1")
BUCKET      = os.getenv("BUCKET", "unit3-capstone")
OS_ENDPOINT = os.getenv("OS_ENDPOINT", "https://search-unit3-capstone-brook-occfu3pblfnlyuaody2frosxlq.us-east-1.es.amazonaws.com")  # trailing slash ok
OS_USER     = os.getenv("OS_USER", "admin")
OS_PASS     = os.getenv("OS_PASS", "TempPassw0rd!")
INDEX       = os.getenv("INDEX", "doc_chunks")

DIM = 1536
MODEL_ID = "amazon.titan-embed-text-v1"

s3 = boto3.client("s3", region_name=REGION)
brt = boto3.client("bedrock-runtime", region_name=REGION)

def chunk_text(txt: str, max_chars=2000, overlap=200):
    txt = txt.strip()
    chunks, start, n = [], 0, len(txt)
    while start < n:
        end = min(start + max_chars, n)
        cut = txt.rfind("\n", start, end)
        if cut == -1 or cut <= start + 500:
            cut = end
        chunk = txt[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(cut - overlap, cut)
    return chunks

def embed(texts):
    vecs = []
    for t in texts:
        body = json.dumps({"inputText": t})
        resp = brt.invoke_model(modelId=MODEL_ID, body=body)
        payload = json.loads(resp["body"].read())
        vecs.append(payload["embedding"])
    return vecs

def ensure_index():
    base = OS_ENDPOINT.rstrip("/")
    head = requests.head(f"{base}/{INDEX}", auth=(OS_USER, OS_PASS), timeout=30)
    if head.status_code == 200:
        print(f"[index] {INDEX} exists")
        return

    mapping = {
        "settings": {
            "index": {"knn": True, "knn.algo_param.ef_search": 100}
        },
        "mappings": {
            "properties": {
                "doc_id":   {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "source_s3":{"type": "keyword"},
                "text":     {"type": "text"},
                "embedding":{"type": "knn_vector",
                             "dimension": DIM,
                             "method": {
                                 "name": "hnsw",
                                 "space_type": "cosinesimil",
                                 "engine": "nmslib",
                                 "parameters": {"m": 16, "ef_construction": 128}
                             }}
            }
        }
    }
    cr = requests.put(f"{base}/{INDEX}", auth=(OS_USER, OS_PASS),
                      json=mapping, timeout=60)
    cr.raise_for_status()
    print(f"[index] created {INDEX}")

def list_processed_txt_keys(bucket):
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": "processed/"}
        if token: kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".txt"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def get_s3_text(bucket, key) -> str:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8", errors="ignore")

def index_doc_chunks(bucket, key):
    base = OS_ENDPOINT.rstrip("/")
    doc_id = os.path.basename(key).replace(".pdf.txt", "")
    text   = get_s3_text(bucket, key)
    if not text.strip():
        print(f"[skip-empty] {key}")
        return 0

    chunks = chunk_text(text)
    vectors = embed(chunks)

    count = 0
    for i, (t, v) in enumerate(zip(chunks, vectors)):
        body = {
            "doc_id": doc_id,
            "chunk_id": i,
            "source_s3": f"s3://{bucket}/{key}",
            "text": t,
            "embedding": v
        }
        r = requests.post(f"{base}/{INDEX}/_doc",
                          auth=(OS_USER, OS_PASS),
                          json=body, timeout=30)
        r.raise_for_status()
        count += 1
    print(f"[indexed] {count} chunks for {doc_id}")
    return count

def knn_search(query: str, k=5):
    base = OS_ENDPOINT.rstrip("/")
    qvec = embed([query])[0]
    body = {
        "size": k,
        "query": { "knn": { "embedding": { "vector": qvec, "k": k } } },
        "_source": ["doc_id","chunk_id","source_s3","text"]
    }
    r = requests.get(f"{base}/{INDEX}/_search",
                     auth=(OS_USER, OS_PASS),
                     json=body, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    print(f"\n=== Top {len(hits)} results for: {query!r} ===")
    for h in hits:
        src = h["_source"]
        print(f"[{h['_score']:.3f}] {src['doc_id']}#{src['chunk_id']}  {src['source_s3']}")
        snippet = src["text"].replace("\n", " ")
        print(snippet[:240], "\n")

def main():
    ensure_index()
    keys = list_processed_txt_keys(BUCKET)
    if not keys:
        print("No processed/*.txt found in S3. Upload a PDF so Lambda creates one, then rerun.")
        return
    total = 0
    for k in keys:
        total += index_doc_chunks(BUCKET, k)
    print(f"\nDone. Indexed {total} chunks.\n")

    #Dummy question for my example doc
    knn_search("How do you use SAL?", k=5)

if __name__ == "__main__":
    main()