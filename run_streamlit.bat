@echo off
REM RCS POD Analysis Streamlit Interface Launcher
echo Starting RCS POD Analysis Dashboard...
echo.

REM 切换到批处理文件所在目录
cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM 检查文件是否存在
if not exist "streamlit_app.py" (
    echo Error: streamlit_app.py not found in current directory
    echo Please ensure you are running this from the correct folder
    pause
    exit /b 1
)

echo Open your browser and navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause