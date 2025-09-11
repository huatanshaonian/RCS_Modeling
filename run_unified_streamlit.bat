@echo off
echo Starting Unified RCS Analysis Dashboard...

REM 设置环境变量
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TCL_LIBRARY=G:\anaconda\envs\RCS_OP1\tcl\tcl8.6
set TK_LIBRARY=G:\anaconda\envs\RCS_OP1\tcl\tk8.6

REM 检查依赖
echo Checking dependencies...
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul || echo Warning: PyTorch not found
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')" 2>nul || echo Warning: Streamlit not found

REM 启动统一Streamlit应用
cd /d "%~dp0"
echo Starting Streamlit server...
streamlit run streamlit_unified.py --server.port 8503 --server.headless false

pause