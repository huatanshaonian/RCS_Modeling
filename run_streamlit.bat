@echo off
REM RCS POD Analysis Streamlit Interface Launcher
echo Starting RCS POD Analysis Dashboard...
echo.

REM 切换到批处理文件所在目录
cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM 初始化conda并激活环境
echo Initializing conda and activating environment: RCS_OP1

REM 首先尝试找到conda的安装路径
set "CONDA_PATH="
if exist "G:\anaconda\Scripts\conda.exe" set "CONDA_PATH=G:\anaconda"
if exist "C:\anaconda3\Scripts\conda.exe" set "CONDA_PATH=C:\anaconda3"
if exist "C:\Miniconda3\Scripts\conda.exe" set "CONDA_PATH=C:\Miniconda3"

if "%CONDA_PATH%"=="" (
    echo Error: Conda installation not found
    echo Please ensure Conda/Anaconda is installed and accessible
    echo Trying to use system Python...
    echo.
) else (
    echo Found conda at: %CONDA_PATH%
    
    REM 添加conda到路径
    set "PATH=%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin;%PATH%"
    
    REM 初始化conda环境变量
    call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"
    
    REM 激活RCS_OP1环境
    call "%CONDA_PATH%\Scripts\activate.bat" RCS_OP1
    if %ERRORLEVEL% neq 0 (
        echo Warning: Failed to activate RCS_OP1 environment
        echo Available environments:
        conda env list
        echo.
        echo Trying to continue with base conda environment...
    ) else (
        echo Successfully activated RCS_OP1 environment
    )
)

REM 显示当前Python环境信息
echo Checking Python environment...
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import sys; print('Python version:', sys.version.split()[0])"
echo.

REM 检查文件是否存在
if not exist "streamlit_app.py" (
    echo Error: streamlit_app.py not found in current directory
    echo Please ensure you are running this from the correct folder
    pause
    exit /b 1
)

REM 检查必要的包是否可用
echo Checking required packages...
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Streamlit not found in current environment
    echo Please install streamlit: conda install streamlit
    pause
    exit /b 1
)

python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Warning: PyTorch not found - Autoencoder features will be disabled
)
echo.

echo Open your browser and navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause