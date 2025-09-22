#!/usr/bin/env python3
import os, time, json, base64, io, zipfile, textwrap, uuid
from pathlib import Path

import boto3, botocore, docker, requests

# ----------------------- Config (todo: env file encouraged) -----------------------
REGION       = "us-east-1"
BUCKET       = "unit3-capstone"    # <- set myself (or if you'd like, put this variable in an .env and load from .env)
PREFIX_IN    = "ingest-in/"               # leave as is. bucket subfolder name for ingest CSV/JSON
PREFIX_PDF   = "ingest-pdf/"              # leave as is. bucket subfolder name for ingest PDFs
INDEX        = "doc_chunks"               # leave as is. name of opensearch index

OS_ENDPOINT  = "https://search-data-eng-opensearch-domain-qepm7pxdjp4f33p3inuuzjkrvy.us-east-1.es.amazonaws.com"
OS_USER      = "admin"
OS_PASS      = "TempPassw0rd!"

REPO_NAME    = "embeddings-service"
APP          = "embeddings-svc"                   # ECS/ALB resource name prefix

EMBED_FN     = "embeddings-caller"
CSVJSON_FN   = "csvjson-to-opensearch"
PDF_FN       = "pdf-textract-to-opensearch"

CONTAINER_PORT = 8080
BATCH_SIZE     = 32

# ----------------------- Helpers -----------------------
session = boto3.Session(region_name=REGION)
ec2  = session.client("ec2")
iam  = session.client("iam")
ecr  = session.client("ecr")
ecs  = session.client("ecs")
elbv2= session.client("elbv2")
logs = session.client("logs")
lam  = session.client("lambda")
sts  = session.client("sts")

ACCOUNT = sts.get_caller_identity()["Account"]
REGISTRY = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
REPO_URI = f"{REGISTRY}/{REPO_NAME}"

def ensure_opensearch_index():
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties":{
            "doc_id":{"type":"keyword"},
            "text":{"type":"text"},
            "vec":{"type":"knn_vector",
                   "dimension":384,
                   "method":{"name":"hnsw","space_type":"cosinesimil","engine":"faiss"}}
        }}
    }
    r = requests.put(f"{OS_ENDPOINT}/{INDEX}", auth=(OS_USER, OS_PASS),
                     headers={"Content-Type":"application/json"}, data=json.dumps(body))
    if r.status_code in (200, 201):
        print(f"[OS] Index created: {INDEX}")
    elif r.status_code == 400 and "resource_already_exists_exception" in r.text:
        print(f"[OS] Index exists: {INDEX}")
    else:
        print("[OS] create index status:", r.status_code, r.text[:400])

def write_embedding_files_if_missing():
    if not Path("server_app.py").exists():
        Path("server_app.py").write_text(textwrap.dedent("""\
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
        """))
    if not Path("Dockerfile.server").exists():
        Path("Dockerfile.server").write_text(textwrap.dedent("""\
            FROM python:3.11-slim
            RUN apt-get update && apt-get install -y --no-install-recommends \
                ca-certificates curl && rm -rf /var/lib/apt/lists/*
            ENV PIP_NO_CACHE_DIR=1
            RUN python -m pip install --upgrade pip && \
                python -m pip install --no-cache-dir --only-binary=:all: sentencepiece==0.2.0 && \
                python -m pip install --no-cache-dir numpy==1.26.4 && \
                python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2 && \
                python -m pip install --no-cache-dir \
                    huggingface_hub==0.14.1 tokenizers==0.13.3 transformers==4.30.2 \
                    sentence-transformers==2.2.2 Flask==3.1.1
            RUN python -c "from sentence_transformers import SentenceTransformer as S; m=S('sentence-transformers/all-MiniLM-L6-v2'); m.save('/opt/model'); print('Cached /opt/model')"
            ENV MODEL_PATH=/opt/model
            WORKDIR /app
            COPY server_app.py /app/server_app.py
            EXPOSE 8080
            CMD ["python","server_app.py"]
        """))

def ecr_login_and_repo():
    try:
        repo = ecr.create_repository(repositoryName=REPO_NAME)["repository"]
        print("[ECR] created ECR repo (for embedding-svc app):", repo["repositoryUri"])
    except ecr.exceptions.RepositoryAlreadyExistsException:
        repo = ecr.describe_repositories(repositoryNames=[REPO_NAME])["repositories"][0]
        print("[ECR] using ECR repo (for embedding-svc app):", repo["repositoryUri"])
    auth = ecr.get_authorization_token()["authorizationData"][0]
    username, password = base64.b64decode(auth["authorizationToken"]).decode().split(":")
    print("logging into docker on ECR with creds:", username, password)
    docker.from_env().login(username=username, password=password, registry=auth["proxyEndpoint"])
    print("[ECR] logging into docker successful~!")
    return repo["repositoryUri"], password

def build_and_push_image(password):
    client = docker.from_env(); api = docker.APIClient()
    local_tag = f"{REPO_NAME}:latest"; remote_tag = f"{REPO_URI}:latest"
    print("[IMG] building Dockerfile.server ()…")
    for ch in api.build(path=".", dockerfile="Dockerfile.server", tag=local_tag, decode=True):
        if ch.get("error"): raise RuntimeError(ch["error"])
        s = ch.get("stream");
        if s: s=s.strip(); 
    img = client.images.get(local_tag); img.tag(REPO_URI, "latest")
    print("[IMG] pushing", remote_tag)
    for ch in client.images.push(REPO_URI, tag="latest", auth_config={"username":"AWS","password":password}, stream=True, decode=True):
        if ch.get("error"): raise RuntimeError(ch["error"])
    print("[IMG] push complete")
    return remote_tag

def ensure_exec_role(name):
    assume = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
    try:
        role = iam.get_role(RoleName=name)["Role"]; created=False
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(assume))["Role"]; created=True
    iam.attach_role_policy(RoleName=name, PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy")
    return role["Arn"], created

def vpc_and_subnets():
    vpc = ec2.describe_vpcs(Filters=[{"Name":"isDefault","Values":["true"]}])["Vpcs"][0]
    subs = ec2.describe_subnets(Filters=[{"Name":"vpc-id","Values":[vpc["VpcId"]]}])["Subnets"]
    return vpc["VpcId"], [s["SubnetId"] for s in subs][:2]

def get_or_create_sg(name, desc, vpc_id, ingress=None):
    r = ec2.describe_security_groups(Filters=[{"Name":"group-name","Values":[name]},{"Name":"vpc-id","Values":[vpc_id]}])
    if r["SecurityGroups"]: return r["SecurityGroups"][0]["GroupId"]
    sg = ec2.create_security_group(GroupName=name, Description=desc, VpcId=vpc_id)["GroupId"]
    if ingress:
        try: ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=ingress)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"]!="InvalidPermission.Duplicate": raise
    return sg

def deploy_fargate(image_uri):
    vpc_id, subnets = vpc_and_subnets()
    alb_sg = get_or_create_sg(f"{APP}-alb-sg","ALB SG",vpc_id,[{"IpProtocol":"tcp","FromPort":80,"ToPort":80,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}])
    task_sg= get_or_create_sg(f"{APP}-task-sg","Task SG",vpc_id,[{"IpProtocol":"tcp","FromPort":CONTAINER_PORT,"ToPort":CONTAINER_PORT,"UserIdGroupPairs":[{"GroupId":alb_sg}]}])

    # ALB/TG
    try:
        alb = elbv2.describe_load_balancers(Names=[f"{APP}-alb"])["LoadBalancers"][0]; created=False
    except botocore.exceptions.ClientError:
        alb = elbv2.create_load_balancer(Name=f"{APP}-alb", Subnets=subnets, SecurityGroups=[alb_sg],
                                         Scheme="internet-facing", Type="application", IpAddressType="ipv4")["LoadBalancers"][0]; created=True
    alb_dns = alb["DNSName"]
    try:
        tg = elbv2.describe_target_groups(Names=[f"{APP}-tg"])["TargetGroups"][0]
    except botocore.exceptions.ClientError:
        tg = elbv2.create_target_group(Name=f"{APP}-tg", Protocol="HTTP", Port=CONTAINER_PORT,
                                       VpcId=vpc_id, TargetType="ip",
                                       HealthCheckProtocol="HTTP", HealthCheckPath="/healthz",
                                       HealthCheckPort=str(CONTAINER_PORT))["TargetGroups"][0]
    tg_arn = tg["TargetGroupArn"]
    ls = elbv2.describe_listeners(LoadBalancerArn=alb["LoadBalancerArn"]).get("Listeners", [])
    if not any(l["Port"]==80 for l in ls):
        elbv2.create_listener(LoadBalancerArn=alb["LoadBalancerArn"], Protocol="HTTP", Port=80,
                              DefaultActions=[{"Type":"forward","TargetGroupArn":tg_arn}])

    # ECS
    try: ecs.create_cluster(clusterName=f"{APP}-cluster")
    except ecs.exceptions.ClusterAlreadyExistsException: pass
    exec_arn,_ = ensure_exec_role(f"{APP}-exec-role")

    # logs
    lg=f"/ecs/{APP}"
    try: logs.create_log_group(logGroupName=lg)
    except logs.exceptions.ResourceAlreadyExistsException: pass

    td = ecs.register_task_definition(
        family=f"{APP}-task", networkMode="awsvpc", requiresCompatibilities=["FARGATE"],
        cpu="512", memory="1024", executionRoleArn=exec_arn,
        containerDefinitions=[{
            "name":APP, "image":image_uri,
            "portMappings":[{"containerPort":CONTAINER_PORT,"protocol":"tcp"}],
            "logConfiguration":{"logDriver":"awslogs","options":{"awslogs-group":lg,"awslogs-region":REGION,"awslogs-stream-prefix":"ecs"}}
        }]
    )["taskDefinition"]["taskDefinitionArn"]

    # service: create or update
    try:
        svc = ecs.create_service(cluster=f"{APP}-cluster", serviceName=f"{APP}-svc",
                                 taskDefinition=td, desiredCount=1, launchType="FARGATE",
                                 networkConfiguration={"awsvpcConfiguration":{"subnets":subnets,"securityGroups":[task_sg],"assignPublicIp":"ENABLED"}},
                                 loadBalancers=[{"targetGroupArn":tg_arn,"containerName":APP,"containerPort":CONTAINER_PORT}])["service"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("InvalidParameterException","ResourceInUseException"):
            ecs.update_service(cluster=f"{APP}-cluster", service=f"{APP}-svc", taskDefinition=td, desiredCount=1)
            svc = ecs.describe_services(cluster=f"{APP}-cluster", services=[f"{APP}-svc"])["services"][0]
        else:
            raise

    print("[ECS] waiting for service stable…")
    ecs.get_waiter("services_stable").wait(cluster=f"{APP}-cluster", services=[f"{APP}-svc"])
    print("[ECS] service stable. ALB:", alb_dns)
    return f"http://{alb_dns}"

def create_lambda_zip(name, code_text, env, timeout=30, memory=256):
    role_name = f"{name}-exec-{uuid.uuid4().hex[:8]}"
    assume = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
    role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))["Role"]
    iam.attach_role_policy(RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    print(f"[LAM] created role {role_name}, waiting for propagation…"); time.sleep(12)

    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf: zf.writestr("lambda_function.py", code_text)
    code_bytes=buf.getvalue()

    for attempt in range(8):
        try:
            fn = lam.create_function(FunctionName=name, Role=role["Arn"], Runtime="python3.11",
                                     Handler="lambda_function.handler", Code={"ZipFile":code_bytes},
                                     Timeout=timeout, MemorySize=memory, Publish=True, Environment={"Variables":env})
            print("[LAM] created", fn["FunctionArn"])
            break
        except botocore.exceptions.ClientError as e:
            if "cannot be assumed" in e.response["Error"]["Message"] and attempt<7:
                time.sleep(6*(attempt+1)); continue
            if e.response["Error"]["Code"]=="ResourceConflictException":
                lam.update_function_code(FunctionName=name, ZipFile=code_bytes, Publish=True)
                print("[LAM] updated code for", name); break
            raise
    return lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]

def make_embeddings_caller(embed_url):
    code = textwrap.dedent("""
    import json, os, urllib.request
    EMBED_URL = os.environ["EMBED_URL"]
    def _post_json(url, payload, timeout=25):
        data=json.dumps(payload).encode("utf-8")
        req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return r.getcode(), r.read().decode("utf-8")
    def handler(event, context):
        print("received:", json.dumps(event))
        texts=event.get("texts"); normalize=event.get("normalize", True)
        if not isinstance(texts,list) or not all(isinstance(t,str) for t in texts):
            return {"error":"Expected {'texts':[str,...]}", "received":event}
        status, body = _post_json(EMBED_URL, {"texts":texts,"normalize":normalize})
        try: j=json.loads(body)
        except: return {"http_status":status,"raw":body}
        return {"http_status":status,"dimension":j.get("dimension"),"vectors":j.get("vectors")}
    """).strip()
    arn = create_lambda_zip(EMBED_FN, code, {"EMBED_URL": embed_url}, timeout=30, memory=256)
    print("[LAM] embeddings-caller =", arn)
    return arn

def s3_permission_then_notify(function_name, suffixes, prefix):
    fn_arn = lam.get_function(FunctionName=function_name)["Configuration"]["FunctionArn"]
    sid = f"{function_name}-s3-{uuid.uuid4().hex[:6]}"
    lam.add_permission(FunctionName=function_name, StatementId=sid, Action="lambda:InvokeFunction",
                       Principal="s3.amazonaws.com", SourceArn=f"arn:aws:s3:::{BUCKET}", SourceAccount=ACCOUNT)
    time.sleep(5)
    cfg = session.client("s3").get_bucket_notification_configuration(Bucket=BUCKET)
    lambdas = cfg.get("LambdaFunctionConfigurations", [])
    # upsert per suffix
    for sfx in suffixes:
        id_=f"{function_name}-{sfx}"
        lambdas=[c for c in lambdas if c.get("Id")!=id_]
        lambdas.append({"Id":id_,"LambdaFunctionArn":fn_arn,"Events":["s3:ObjectCreated:*"],
                        "Filter":{"Key":{"FilterRules":[{"Name":"prefix","Value":prefix},{"Name":"suffix","Value":sfx}]}}})
    session.client("s3").put_bucket_notification_configuration(Bucket=BUCKET, NotificationConfiguration={"LambdaFunctionConfigurations":lambdas})
    print(f"[S3] notifications set for {function_name} on s3://{BUCKET}/{prefix} for {', '.join(suffixes)}")

def make_csvjson_lambda():
    code = textwrap.dedent("""
        import os, json, csv, io, urllib.request, urllib.parse, base64

        EMBED_FN    = os.environ["EMBED_FN"]
        OS_ENDPOINT = os.environ["OS_ENDPOINT"].rstrip("/")
        OS_USER     = os.environ["OS_USER"]
        OS_PASS     = os.environ["OS_PASS"]
        INDEX       = os.environ["INDEX"]
        BATCH_SIZE  = int(os.environ.get("BATCH_SIZE","32"))

        def _lambda_invoke(fn_name, payload, timeout=25):
            import boto3
            lam = boto3.client("lambda")
            resp = lam.invoke(FunctionName=fn_name, Payload=json.dumps(payload).encode("utf-8"))
            body = resp["Payload"].read().decode("utf-8","replace")
            if resp.get("StatusCode", 500) >= 300:
                raise RuntimeError("Invoke {} failed: {} {}".format(fn_name, resp.get("StatusCode"), body))
            return json.loads(body)

        def _http_post(url, data_bytes, user, pwd, ctype="application/x-ndjson", timeout=25):
            req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": ctype})
            auth = base64.b64encode("{}:{}".format(user, pwd).encode()).decode()
            req.add_header("Authorization", "Basic {}".format(auth))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.getcode(), r.read().decode("utf-8","replace")

        def _get_s3_object(Bucket, Key, byte_limit=20*1024*1024):
            import boto3
            s3 = boto3.client("s3")
            r = s3.get_object(Bucket=Bucket, Key=Key)
            body = r["Body"].read(byte_limit+1)
            if body is None:
                body = b""
            if len(body) > byte_limit:
                raise RuntimeError("Object too large for this demo ({} bytes > {}).".format(len(body), byte_limit))
            return body

        def _iter_csv_chunks(raw_bytes):
            text = (raw_bytes or b"").decode("utf-8", "replace")
            buf = io.StringIO(text)
            try:
                rows = list(csv.DictReader(buf))
                if rows and any(rows[0]):
                    for row in rows:
                        yield json.dumps(row, ensure_ascii=False)
                    return
            except Exception:
                pass
            buf.seek(0)
            for row in csv.reader(buf):
                yield " ".join([c for c in row if c is not None])

        def _iter_json_chunks(raw_bytes):
            text = (raw_bytes or b"").decode("utf-8","replace").strip()
            if not text:
                return
            if "\\n" in text and text.lstrip().startswith("{"):
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield json.dumps(obj, ensure_ascii=False)
                    except Exception:
                        yield line
                return
            try:
                obj = json.loads(text)
            except Exception:
                yield text
                return
            if isinstance(obj, list):
                for item in obj:
                    yield json.dumps(item, ensure_ascii=False)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, list):
                        for item in v:
                            yield json.dumps(item, ensure_ascii=False)
                        return
                yield json.dumps(obj, ensure_ascii=False)
            else:
                yield str(obj)

        def _bulk_index(doc_id, texts, vectors):
            lines = []
            for i, (t, v) in enumerate(zip(texts, vectors)):
                _id = "{}::{:06d}".format(doc_id, i)
                lines.append(json.dumps({"index":{"_index": INDEX, "_id": _id}}))
                lines.append(json.dumps({"doc_id": doc_id, "text": t, "vec": v}))
            payload = ("\\n".join(lines) + "\\n").encode("utf-8")
            code, body = _http_post("{}/_bulk".format(OS_ENDPOINT), payload, OS_USER, OS_PASS, "application/x-ndjson", timeout=30)
            if code >= 300:
                raise RuntimeError("Bulk index failed: {} {}".format(code, body[:400]))
            res = json.loads(body)
            if res.get("errors"):
                raise RuntimeError("Bulk index reported errors: {}".format(body[:400]))
            return res

        def handler(event, context):
            for rec in event.get("Records", []):
                b = rec["s3"]["bucket"]["name"]
                k = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
                print("Processing s3://{}/{}".format(b, k))
                raw = _get_s3_object(Bucket=b, Key=k)

                lower = k.lower()
                if lower.endswith(".csv"):
                    chunks_iter = _iter_csv_chunks(raw)
                elif lower.endswith(".json"):
                    chunks_iter = _iter_json_chunks(raw)
                else:
                    print("Skipping unsupported suffix for {}".format(k))
                    continue

                texts = list(chunks_iter)
                print("Parsed {} chunks from {}".format(len(texts), k))
                doc_id = "{}".format(k)

                start = 0
                while start < len(texts):
                    batch = texts[start:start+BATCH_SIZE]
                    start += BATCH_SIZE
                    emb = _lambda_invoke(EMBED_FN, {"texts": batch, "normalize": True})
                    if emb.get("http_status") != 200:
                        raise RuntimeError("Embedding call failed: {}".format(emb))
                    vecs = emb.get("vectors") or []
                    if len(vecs) != len(batch):
                        raise RuntimeError("Embedding count mismatch: got {} for {} texts".format(len(vecs), len(batch)))
                    res = _bulk_index(doc_id, batch, vecs)
                    print("Indexed batch of {} (took={})".format(len(batch), res.get("took")))

            return {"ok": True}
    """).strip()

    # role & inline perms
    role = iam.create_role(
        RoleName=f"{CSVJSON_FN}-exec-{uuid.uuid4().hex[:8]}",
        AssumeRolePolicyDocument=json.dumps({
            "Version":"2012-10-17",
            "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]
        })
    )["Role"]
    iam.attach_role_policy(
        RoleName=role["RoleName"],
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    )
    inline = {
        "Version":"2012-10-17",
        "Statement":[
            {"Effect":"Allow","Action":["s3:GetObject"],"Resource":[f"arn:aws:s3:::{BUCKET}/{PREFIX_IN}*"]},
            {"Effect":"Allow","Action":["lambda:InvokeFunction"],"Resource":[f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:{EMBED_FN}"]}
        ]
    }
    iam.put_role_policy(RoleName=role["RoleName"], PolicyName="S3ReadAndInvokeEmbed", PolicyDocument=json.dumps(inline))
    time.sleep(12)

    # package & create/update
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", code)

    try:
        lam.create_function(
            FunctionName=CSVJSON_FN,
            Role=role["Arn"],
            Runtime="python3.11",
            Handler="lambda_function.handler",
            Code={"ZipFile":buf.getvalue()},
            Timeout=120,
            MemorySize=512,
            Environment={"Variables":{
                "EMBED_FN":EMBED_FN,
                "OS_ENDPOINT":OS_ENDPOINT,
                "OS_USER":OS_USER,
                "OS_PASS":OS_PASS,
                "INDEX":INDEX,
                "BATCH_SIZE":str(BATCH_SIZE)
            }},
            Publish=True
        )
        print("[LAM] created", CSVJSON_FN)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"]=="ResourceConflictException":
            lam.update_function_code(FunctionName=CSVJSON_FN, ZipFile=buf.getvalue(), Publish=True)
            print("[LAM] updated code for", CSVJSON_FN)
        else:
            raise

    # permission then S3 notifications
    s3_permission_then_notify(CSVJSON_FN, [".csv",".json"], PREFIX_IN)

def make_pdf_lambda():
    code = textwrap.dedent("""
    import os, json, time, base64, urllib.request, urllib.parse, boto3
    EMBED_FN=os.environ["EMBED_FN"]; OS_ENDPOINT=os.environ["OS_ENDPOINT"].rstrip("/")
    OS_USER=os.environ["OS_USER"]; OS_PASS=os.environ["OS_PASS"]; INDEX=os.environ["INDEX"]
    BATCH_SIZE=int(os.environ.get("BATCH_SIZE","24")); POLL_SEC=int(os.environ.get("POLL_SEC","2")); MAX_POLLS=int(os.environ.get("MAX_POLLS","120"))
    def _emb(texts):
        lam=boto3.client("lambda"); r=lam.invoke(FunctionName=EMBED_FN,Payload=json.dumps({"texts":texts,"normalize":True}).encode("utf-8"))
        body=r["Payload"].read().decode("utf-8","replace"); out=json.loads(body)
        if out.get("http_status")!=200: raise RuntimeError(f"embed http {out.get('http_status')}: {body[:200]}")
        return out["vectors"]
    def _post_bulk(lines):
        req=urllib.request.Request(f"{OS_ENDPOINT}/_bulk",data=("\\n".join(lines)+"\\n").encode("utf-8"),headers={"Content-Type":"application/x-ndjson"})
        auth=base64.b64encode(f"{OS_USER}:{OS_PASS}".encode()).decode(); req.add_header("Authorization", f"Basic {auth}")
        with urllib.request.urlopen(req,timeout=30) as r: return r.getcode(), r.read().decode("utf-8","replace")
    def _chunk(lines, max_chars=1800, max_lines=70):
        chunks=[]; cur=[]; ln=0
        for t in lines:
            t=t.strip(); if not t: continue
            if ln+len(t)+1>max_chars or len(cur)>=max_lines:
                if cur: chunks.append(" ".join(cur)); cur=[]; ln=0
            cur.append(t); ln+=len(t)+1
        if cur: chunks.append(" ".join(cur))
        return chunks
    def _textract_lines(b,k):
        tex=boto3.client("textract"); job=tex.start_document_text_detection(DocumentLocation={"S3Object":{"Bucket":b,"Name":k}})["JobId"]
        lines=[]; tok=None; polls=0
        while True:
            args={"JobId":job}; 
            if tok: args["NextToken"]=tok
            r=tex.get_document_text_detection(**args)
            st=r["JobStatus"]
            if st=="SUCCEEDED":
                for bl in r.get("Blocks",[]):
                    if bl.get("BlockType")=="LINE" and "Text" in bl: lines.append(bl["Text"])
                tok=r.get("NextToken"); 
                if not tok: break
            elif st in ("FAILED","PARTIAL_SUCCESS"): raise RuntimeError(f"textract {st}")
            else:
                polls+=1
                if polls>MAX_POLLS: raise RuntimeError("textract timeout")
                time.sleep(POLL_SEC)
        return lines
    def handler(event, context):
        for rec in event.get("Records",[]):
            b=rec["s3"]["bucket"]["name"]; k=urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
            if not k.lower().endswith(".pdf"): continue
            lines=_textract_lines(b,k); chunks=_chunk(lines); doc_id=k; start=0
            while start<len(chunks):
                batch=chunks[start:start+BATCH_SIZE]; start+=BATCH_SIZE
                vecs=_emb(batch)
                lines=[]
                for i,(t,v) in enumerate(zip(batch,vecs)):
                    _id=f"{doc_id}::{i:06d}"; lines.append(json.dumps({"index":{"_index":INDEX,"_id":_id}}))
                    lines.append(json.dumps({"doc_id":doc_id,"text":t,"vec":v}))
                code,body=_post_bulk(lines)
                if code>=300: raise RuntimeError(body[:300])
        return {"ok":True}
    """).strip()

    role = iam.create_role(RoleName=f"{PDF_FN}-exec-{uuid.uuid4().hex[:8]}",
                           AssumeRolePolicyDocument=json.dumps({"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}))["Role"]
    iam.attach_role_policy(RoleName=role["RoleName"], PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    inline={"Version":"2012-10-17","Statement":[
        {"Effect":"Allow","Action":["s3:GetObject"],"Resource":[f"arn:aws:s3:::{BUCKET}/{PREFIX_PDF}*"]},
        {"Effect":"Allow","Action":["textract:StartDocumentTextDetection","textract:GetDocumentTextDetection"],"Resource":"*"},
        {"Effect":"Allow","Action":["lambda:InvokeFunction"],"Resource":[f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:{EMBED_FN}"]}
    ]}
    iam.put_role_policy(RoleName=role["RoleName"], PolicyName="TextractReadInvoke", PolicyDocument=json.dumps(inline))
    time.sleep(12)

    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf: zf.writestr("lambda_function.py", code)
    try:
        lam.create_function(FunctionName=PDF_FN, Role=role["Arn"], Runtime="python3.11", Handler="lambda_function.handler",
                            Code={"ZipFile":buf.getvalue()}, Timeout=180, MemorySize=1024,
                            Environment={"Variables":{"EMBED_FN":EMBED_FN,"OS_ENDPOINT":OS_ENDPOINT,"OS_USER":OS_USER,"OS_PASS":OS_PASS,"INDEX":INDEX,"BATCH_SIZE":"24","POLL_SEC":"2","MAX_POLLS":"120"}},
                            Publish=True)
        print("[LAM] created", PDF_FN)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"]=="ResourceConflictException":
            lam.update_function_code(FunctionName=PDF_FN, ZipFile=buf.getvalue(), Publish=True)
            print("[LAM] updated code for", PDF_FN)
        else: raise

    s3_permission_then_notify(PDF_FN, [".pdf"], PREFIX_PDF)

# ----------------------- main -----------------------
def main():
    # 0) Pre-flight
    #assert BUCKET and BUCKET!="YOUR-BUCKET-NAME", "Set LAB_BUCKET (or edit BUCKET) before running."
    write_embedding_files_if_missing()

    # 1) OpenSearch index
    ensure_opensearch_index()

    # 2) Build + push embedding image
    repo_uri, pw = ecr_login_and_repo()
    image_uri = build_and_push_image(pw)

    # 3) Fargate service + ALB
    embed_url = deploy_fargate(image_uri) + "/embed"

    # 4) embeddings-caller Lambda
    make_embeddings_caller(embed_url)

    # 5) csv/json ingestor Lambda + S3 notifications
    make_csvjson_lambda()

    # 6) pdf (Textract) ingestor Lambda + S3 notifications
    make_pdf_lambda()

    # 7) Print quick tests
    print("\n=== Quick command line tests you can run to test ingestion > chunking (for pdfs) > embedding: ===")
    print(f"Make a CSV and upload it:   echo 'a,b,c\\nx,y,z' | aws s3 cp - s3://{BUCKET}/{PREFIX_IN}demo.csv")
    print(f"Make a JSON and upload it:  echo '[{{\"msg\":\"hello\"}},{{\"msg\":\"world\"}}]' | aws s3 cp - s3://{BUCKET}/{PREFIX_IN}demo.json")
    print(f"Upload your own sample PDF:   aws s3 cp sample.pdf s3://{BUCKET}/{PREFIX_PDF}sample.pdf")
    print("\nOpenSearch endpoint:", OS_ENDPOINT)
    print("OpenSearch Index name:", INDEX)
    print("Done ✅")

if __name__ == "__main__":
    main()