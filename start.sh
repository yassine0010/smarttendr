#!/bin/bash
cd "/Users/yassinebenayed/smart tender"

# Start FastAPI in background
./venv/bin/python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Start Streamlit in background
./venv/bin/python -m streamlit run frontend/app.py --server.port 8501 --server.headless true &
ST_PID=$!

echo ""
echo "========================================="
echo "  SmartTender AI - Servers Running"
echo "========================================="
echo "  API:       http://localhost:8000"
echo "  Dashboard: http://localhost:8501"
echo "  API PID:   $API_PID"
echo "  Streamlit: $ST_PID"
echo "========================================="
echo "  Press Ctrl+C to stop both servers"
echo "========================================="

trap "kill $API_PID $ST_PID 2>/dev/null; exit" INT TERM
wait
