@echo off
REM RCS POD Analysis Streamlit Interface Launcher
echo Starting RCS POD Analysis Dashboard...
echo.
echo Open your browser and navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause