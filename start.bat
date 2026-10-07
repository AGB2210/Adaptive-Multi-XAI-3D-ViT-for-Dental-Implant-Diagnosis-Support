@echo off
rem ===========================================================================
rem  Implant Site Screening -- one click.
rem
rem  Finds Python, sets up the environment the first time, starts the server
rem  and opens the page in the browser. Ctrl+C, or closing this window, stops
rem  the server and frees its port.
rem
rem  Anything after the file name is passed on to `python -m app`:
rem      start.bat --port 9000 --device cpu
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

rem After Ctrl+C cmd asks "Terminate batch job (Y/N)?" and waits for a key. The
rem server has already stopped by then, but the window would sit on a question
rem nobody needs to answer. Running the script again with its input taken from
rem NUL answers the question at once.
if defined IMPLANT_APP_LAUNCHED goto :main
set "IMPLANT_APP_LAUNCHED=1"
call <nul "%~f0" %*
exit /b

:main
title Implant Site Screening

rem ---- find Python: this folder's own environment first, then the system's --
set "PY="
if exist ".venv\Scripts\python.exe" set PY=".venv\Scripts\python.exe"
if not defined PY call :probe python
if not defined PY call :probe py -3
if not defined PY goto :no_python

:run
%PY% -m app --open %*
set "CODE=%errorlevel%"
rem 3 is the app saying a package is missing. -1073741510 is Windows' own code
rem for a process ended by Ctrl+C, which is a normal way to stop.
if "%CODE%"=="3" if not defined SETUP_DONE goto :setup
if "%CODE%"=="0" exit /b 0
if "%CODE%"=="-1073741510" exit /b 0
echo.
echo  The app stopped with an error ^(exit code %CODE%^). The message above says why.
goto :hold

rem ---- first run: an environment in .venv, with only what is missing ---------
:setup
set "SETUP_DONE=1"
echo.
echo  The app's Python packages are not installed yet. Setting them up in .venv.
echo  This happens once. If PyTorch has to be downloaded it takes several minutes.
echo.
if exist ".venv\Scripts\python.exe" goto :packages
rem --system-site-packages: a PyTorch already installed for this Python is used
rem as it is, instead of downloading a second copy of it.
%PY% -m venv --system-site-packages .venv
if errorlevel 1 goto :setup_failed

:packages
set PY=".venv\Scripts\python.exe"
%PY% -m pip install --upgrade pip
%PY% -c "import torch" >nul 2>nul
if not errorlevel 1 goto :requirements
rem PyTorch comes first and from its own index: the build on PyPI has no GPU
rem support on Windows. Set TORCH_INDEX_URL beforehand to choose another build.
if defined TORCH_INDEX_URL goto :torch
nvidia-smi >nul 2>nul
if errorlevel 1 goto :requirements
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126"

:torch
echo  Installing PyTorch from %TORCH_INDEX_URL%
%PY% -m pip install torch --index-url "%TORCH_INDEX_URL%"
if errorlevel 1 echo  That build did not install; the default PyTorch build will be used.

:requirements
%PY% -m pip install -r requirements-app.txt
if errorlevel 1 goto :setup_failed
echo.
goto :run

:probe
rem Sets PY to the command given if it is Python 3.11 or newer. Run rather than
rem looked up: the `python` Windows ships is a stub that opens the Store.
%* -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>nul
if not errorlevel 1 set PY=%*
exit /b 0

:no_python
echo.
echo  Python 3.11 or newer was not found.
echo  Install it from https://www.python.org/downloads/ and tick "Add python.exe
echo  to PATH" in the installer, then run start.bat again.
goto :hold

:setup_failed
echo.
echo  Setting up the Python environment failed. The messages above say why.
echo  Fix that and run start.bat again; what was already installed is kept.

:hold
echo.
echo  Press any key to close this window.
pause <con >nul
exit /b 1
