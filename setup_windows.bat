@echo off
setlocal

set "PYTHON_CMD=py -3.10"
if exist .python310\cpython-3.10.21-windows-x86_64-none\python.exe (
    set "PYTHON_CMD=.python310\cpython-3.10.21-windows-x86_64-none\python.exe"
) else (
    py -3.10 --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.10 is required. Install it and make sure "py -3.10" works.
        exit /b 1
    )
)

if not exist .venv\Scripts\python.exe (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    .venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
) else (
    .venv\Scripts\python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
)
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
echo Setup finished. Activate with: .venv\Scripts\activate
