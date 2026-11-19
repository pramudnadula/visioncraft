# Storage & Database Configuration Guide

FaceVision uses **pluggable storage and database adapters** so any user can run 100% locally or connect free-tier / self-hosted cloud services via `.env`.

---

## 1. Storage Providers (`FACEVISION_STORAGE_BACKEND`)

| Provider | Setting | Notes |
| :--- | :--- | :--- |
| **Local Disk / Local Server** (Default) | `local` | Zero config; stores files under `.facevision_data/` with UUID keys. |
| **Cloudflare R2** (Free 10 GB tier) | `s3` | Set `FACEVISION_S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com` and `FACEVISION_S3_REGION=auto`. |
| **Backblaze B2 / Supabase S3 / MinIO** | `s3` | Set `FACEVISION_S3_ENDPOINT_URL` to your provider's S3 endpoint + bucket/keys. |
| **AWS S3** | `s3` | Leave `FACEVISION_S3_ENDPOINT_URL` empty and set bucket, region, and keys. |
| **Firebase Storage / GCS** (Free Spark/Always Free tier) | `firebase` or `gcs` | Set `FACEVISION_GCS_BUCKET=<project>.appspot.com` and `FACEVISION_FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccount.json`. |

*(Install optional cloud SDKs with `pip install -e ".[cloud]"`.)*

---

## 2. Database Providers (`FACEVISION_DATABASE_URL`)

FaceVision uses **SQLAlchemy 2.0**, supporting:
- **SQLite** (Default): Zero setup, stored at `.facevision_data/facevision.db`.
- **PostgreSQL** (Free tiers: **Neon**, **Supabase**, **Render**, **Railway**, or Docker):
  ```env
  FACEVISION_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME
  ```
- **MySQL / MariaDB**:
  ```env
  FACEVISION_DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:3306/DBNAME
  ```
