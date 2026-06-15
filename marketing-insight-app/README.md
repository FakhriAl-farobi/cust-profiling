# Artajasa Marketing Insight App

React + Tailwind CSS frontend with a FastAPI backend that ports the Streamlit customer profiling prototype.

## Run Backend

```powershell
cd "C:\Users\ASUS\Desktop\Program\Artajasa\marketing-insight-app\backend"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Optional: copy `.env.example` to `.env` and set `DATABASE_URL` / `GEMINI_API_KEY`.

## Run Frontend

```powershell
cd "C:\Users\ASUS\Desktop\Program\Artajasa\marketing-insight-app\frontend"
npm.cmd run dev -- --host 127.0.0.1 --port 5174
```

Open `http://127.0.0.1:5174`.

## Notes

- The API tries PostgreSQL table `data_transaksi_qris` first.
- If the database is unavailable, it falls back to `dataset/market_analyst_12000_transactions.csv` so the UI remains testable.
- Main endpoints: `/health`, `/categories`, `/insights`, `/cluster`, `/chat`.
