# This directory is the shared volume between n8n and the Forecast API containers.
#
# Structure:
#   shared/
#   ├── uploads/     ← CSV files received from Gmail (written by Flask API)
#   └── outputs/     ← PNG chart images (written by Flask API, read by n8n)
#
# Both containers mount this folder:
#   - forecast_api: /shared
#   - n8n:          /shared
#
# Do NOT delete this folder — Docker volume mount needs it to exist.
