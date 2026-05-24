# 🛒 Retail Sales Forecasting — N8N Gmail Automation

> **Created by: sanT**  
> Automates the ML forecasting pipeline via Gmail + n8n + Docker.  
> Send a CSV → Get forecast results + charts back by email. Automatically.

---

## 📁 Folder Structure

```
n8n_automation/
├── docker-compose.yml           # Orchestrates n8n + Flask API
├── api/
│   ├── Dockerfile               # Python 3.11 + full ML stack
│   ├── requirements.txt         # All Python dependencies
│   ├── forecast_api.py          # Flask REST API (wraps main.py)
│   └── shared/
│       ├── uploads/             # Received CSV files
│       └── outputs/             # Generated chart PNGs
├── n8n_workflows/
│   └── gmail_forecast_workflow.json   # Import this into n8n
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker Desktop | Latest | With WSL2 backend (Windows) |
| Gmail Account | Any | OAuth2 credentials needed |
| Google Cloud Project | Any | Free tier is fine |

---

### Step 1 — Start the Stack

1. **Configure Environment Variables**:  
   Copy the template provided in the root directory:
   ```powershell
   cp ../.env.example ../.env
   ```
   Open `../.env` and update the `N8N_BASIC_AUTH_PASSWORD` and other credentials.

2. **Run with Docker Compose**:
   ```powershell
   # Navigate to this folder
   cd d:\final\n8n_automation

   # Start both services (n8n + Flask API)
   docker compose up -d

   # Check logs
   docker compose logs -f
   ```

Services will be available at:
- **n8n UI**: http://localhost:5678 
- **Flask API**: http://localhost:8000

> ⚠️ **Important**: All secrets are now managed in the root `.env` file. Do not commit this file to GitHub!

---

### Step 2 — Set Up Gmail OAuth2 Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable **Gmail API**: APIs & Services → Enable APIs → Search "Gmail API" → Enable
4. Create OAuth2 credentials:
   - Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
   - Application type: **Web application**
   - Authorized redirect URIs: `http://localhost:5678/rest/oauth2-credential/callback`
5. Download the JSON — note the **Client ID** and **Client Secret**

**In n8n UI:**
1. Go to **Settings** → **Credentials** → **Add Credential**
2. Search for **Gmail OAuth2**
3. Enter your Client ID and Client Secret
4. Click **Connect** → authorize with your Google account
5. **Save** the credential

---

### Step 3 — Import the Workflow

1. Open n8n at http://localhost:5678
2. Click **Workflows** → **Import from File**
3. Select `n8n_workflows/gmail_forecast_workflow.json`
4. In the imported workflow, click each **Gmail** node and update the credential:
   - Click the node → **Credentials** → select your Gmail OAuth2 credential
5. **Save** and **Activate** the workflow (toggle at top right)

---

### Step 4 — Send Your First Forecast

Send an email to your connected Gmail account:

```
Subject: [FORECAST] or forecast or FORECAST 
Attachment: your_sales_data.csv
Body: (optional — any text)
```

**Your CSV needs:**
- A **date column** (any name containing "date", "time", "period", etc.)
- A **numeric sales/revenue column** (any name containing "sales", "revenue", "amount", etc.)
- At least **20 rows** of data

Within **2–10 minutes** you'll receive a reply with:
- 📊 Full forecast summary (best model, RMSE, predictions)
- 📋 Model leaderboard (all trained models ranked)
- 🖼️ Attached chart images (forecast plot, model comparison, error distribution)

---

## ⚙️ Advanced Options

### Custom Mode & Periods via Subject Line

```
[FORECAST] or forecast             → Quick mode, 30 periods (default)
[FORECAST:full:60] or forecast:full:60      → Full mode (22 models), 60 periods ahead
[FORECAST:quick:90] or forecast:quick:90     → Quick mode, 90 periods ahead
[FORECAST:full:365] or forecast:full:365     → Full mode, 365 periods ahead (max)
```

### Test the API Directly

```powershell
# Health check
curl http://localhost:8000/health

# Run a forecast manually (test)
curl -X POST http://localhost:8000/forecast `
  -F "file=@C:\path\to\your\data.csv" `
  -F "mode=quick" `
  -F "periods=30"
```

---

## 🔗 Workflow Trigger

| Trigger | How to Use | Purpose |
|---------|-----------|---------|
| **Gmail Trigger** | Send email with `[FORECAST]` subject + CSV | Main automation |

---


---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
│                                                                 │
│   ┌──────────────────┐          ┌──────────────────────────┐   │
│   │   n8n Container  │          │  Forecast API Container  │   │
│   │   port: 5678     │  HTTP    │  port: 8000              │   │
│   │                  │◄────────►│                          │   │
│   │  Gmail Trigger   │          │  POST /forecast          │   │
│   │  IF Node         │          │  GET  /outputs/<id>/<f>  │   │
│   │  HTTP Request    │          │  GET  /health            │   │
│   │  Gmail Send      │          │                          │   │
│   └──────────────────┘          └──────────────┬───────────┘   │
│           │                                    │               │
│           └──────────────┬─────────────────────┘               │
│                     Shared Volume                               │
│                   ./api/shared/                                 │
│                   ├── uploads/  (CSVs)                          │
│                   └── outputs/  (PNG charts)                    │
└─────────────────────────────────────────────────────────────────┘
         │                           │
    Gmail Inbox                d:\final\ (bind mount)
    (User's CSV)               (your existing code)
```

---

## 🔧 Configuration

Key settings in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `N8N_BASIC_AUTH_USER` | `admin` | n8n UI username |
| `N8N_BASIC_AUTH_PASSWORD` | `forecast2024` | n8n UI password — **CHANGE THIS** |
| `GENERIC_TIMEZONE` | `Asia/Kolkata` | Your timezone |


---

## 🐛 Troubleshooting

### API not healthy?
```powershell
docker compose logs forecast_api
```
Common issue: project path bind mount. Check `docker-compose.yml` → `volumes: - ../:/app/project:ro`

### Gmail not connecting?
- Ensure OAuth redirect URI is exactly: `http://localhost:5678/rest/oauth2-credential/callback`
- Make sure Gmail API is enabled in Google Cloud Console

### Pipeline fails on CSV?
- The API returns detailed error in email and logs
- Check: `docker compose logs forecast_api`
- Your CSV needs 20+ rows after date aggregation

### Charts not attaching?
- Check shared volume: `ls d:\final\n8n_automation\api\shared\outputs\`
- n8n reads from `/shared/outputs/<run_id>/` inside container

---

## 📦 Stopping / Restarting

```powershell
# Stop
docker compose down

# Restart
docker compose up -d

# Rebuild API after code changes
docker compose up -d --build forecast_api

# View all logs
docker compose logs -f

# View specific service logs
docker compose logs forecast_api -f
docker compose logs n8n -f
```

---

## 🔐 Security Notes

- This setup is for **local use only** by default
- For production/cloud deployment, use HTTPS and proper secrets management
- Never commit Gmail OAuth credentials to git
- Change the n8n basic auth password

---

*Powered by [n8n](https://n8n.io/) + [Flask](https://flask.palletsprojects.com/) + Future Retail Sales Forecasting v2.0*  
*Created by sanT*
