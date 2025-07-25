@echo off
REM ===============================
REM Floor Plan Project Setup Script
REM ===============================

REM Step 1: Install Python if not installed
echo Checking Python installation...
python --version > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Installing Python 3.10.11...
    curl -L -o python-installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python-installer.exe
    echo Python installed successfully.
) ELSE (
    echo Python is already installed.
)

REM Step 2: Install Git if not installed
echo Checking Git installation...
git --version > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Installing Git...
    curl -L -o git-installer.exe https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe
    start /wait git-installer.exe /VERYSILENT
    del git-installer.exe
    echo Git installed successfully.
) ELSE (
    echo Git is already installed.
)

REM Step 3: Clone the GitHub repository or initialize if not present
set /p REPO_URL="Enter the GitHub repository URL (or press Enter to skip cloning): "
IF NOT "%REPO_URL%"=="" (
    echo Cloning the repository: %REPO_URL%
    git clone %REPO_URL%
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to clone repository. Exiting.
        exit /b 1
    )
    for %%d in (%REPO_URL%) do set REPO_DIR=%%~nxd
    echo Entering repository directory: %REPO_DIR%
    cd %REPO_DIR%
) ELSE (
    echo No repository URL provided. Using current directory.
    IF NOT EXIST .git (
        echo Initializing a new git repository...
        git init
    )
)

REM Step 4: Set up a virtual environment
echo Setting up virtual environment...
python -m venv venv
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    call venv\bin\activate
)

REM Step 5: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install ultralytics torch torchvision torchaudio opencv-python roboflow transformers segment-anything

REM Step 6: Download YOLOv8 model weights if not present
if not exist yolov8l.pt (
    echo Downloading yolov8l.pt model weights...
    curl -L -o yolov8l.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
) else (
    echo yolov8l.pt already exists.
)
REM (Optional) Download yolov8x.pt if you want to use the extra large model
REM if not exist yolov8x.pt (
REM     echo Downloading yolov8x.pt model weights...
REM     curl -L -o yolov8x.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
REM )

REM Step 7: Download SAM model weights (manual step if not automated)
echo Please download the Segment Anything Model (SAM) weights manually if required.
echo See: https://github.com/facebookresearch/segment-anything#model-checkpoints

REM Step 8: Install linting and code quality tools
echo Installing linting and code quality tools...
pip install flake8 black isort pre-commit

REM Step 9: Set up pre-commit hooks (optional)
if exist .pre-commit-config.yaml (
    echo Installing pre-commit hooks...
    pre-commit install
)

REM Step 10: Verify installation
echo Verifying installation...
python --version
git --version
pip list

REM Step 11: Run the test script (optional)
echo Running the test script...
if exist train.py (
    python train.py
) else (
    echo No train.py script found. Setup complete.
)

echo.
echo ===============================
echo Setup complete! You are ready to go.
echo =============================== 