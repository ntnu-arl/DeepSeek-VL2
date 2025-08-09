#!/bin/bash

# ===== CONFIG ===== #
export API_KEY_VALUE="arl-vlm"
HOST="0.0.0.0"
PORT=8000
SCRIPT_PATH="main:app"  # Adjust if your app is elsewhere
TUNNEL_NAME="vlm-tunnel"
CLOUDFLARED_CONFIG=""  # Optional custom config
ENV_PATH="/home/ubuntu/vlm_server_env/bin/python"
export USE_CUDA="True"  # Set to "true" if you want to use CUDA

# ===== SET ENVIRONMENT VARIABLE ===== #
export API_KEY="$API_KEY_VALUE"
echo "[INFO] API_KEY set"

# ===== RUN UVICORN BACKEND ===== #
echo "[INFO] Starting FastAPI server on http://$HOST:$PORT ..."
$ENV_PATH -m uvicorn "$SCRIPT_PATH" --host "$HOST" --port "$PORT" &
UVICORN_PID=$!

# ===== START CLOUDFLARED TUNNEL ===== #
sleep 2  # Ensure uvicorn starts first

# if [ -f "$CLOUDFLARED_CONFIG" ]; then
#     echo "[INFO] Using cloudflared config: $CLOUDFLARED_CONFIG"
#     cloudflared tunnel --config "$CLOUDFLARED_CONFIG" run "$TUNNEL_NAME"
# else
#     echo "[WARN] No config.yml found — using default tunnel to port $PORT"
#     cloudflared tunnel --url http://localhost:$PORT --no-autoupdate
# fi

# ===== CLEANUP ===== #
trap "echo '[INFO] Shutting down server...'; kill $UVICORN_PID" EXIT
wait $UVICORN_PID
