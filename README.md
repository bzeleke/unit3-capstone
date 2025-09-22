Process flow diagram:

Upload to S3 (PDF, CSV, JSON)

S3 Bucket: unit3-capstone
-ingest pdf / ingest csv+json

Lambda function:
  -processing docs
  -trigger:s3 upload

Output written to s3 when analyzed by textract

ingest.py reads processed file and sends chunks
to open search
