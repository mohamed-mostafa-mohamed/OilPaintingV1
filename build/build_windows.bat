@echo off
setlocal
rem Builds dist\OilStudio\OilStudio.exe and dist\OilStudio-windows.zip.
rem Installs everything it needs: Python 3.12 (if missing), the WebView2
rem runtime (if missing), then all the Python packages.
rem Run dist\OilStudio\OilStudio.exe - never an .exe outside its _internal folder.
cd /d "%~dp0.."
set PY_VERSION=3.12.10
set PY_LOCAL=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

rem ---- 1. Python --------------------------------------------------------
set PY=
python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1 && set PY=python
if not defined PY py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1 && set PY=py -3
if not defined PY if exist "%PY_LOCAL%" set PY="%PY_LOCAL%"
if not defined PY (
  echo == Python not found - downloading Python %PY_VERSION% from python.org
  powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe' -OutFile \"$env:TEMP\python-installer.exe\""
  if errorlevel 1 (echo Could not download Python. Check the internet connection. & pause & exit /b 1)
  echo == Installing Python for this user ^(no admin needed^)
  "%TEMP%\python-installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0 Include_test=0
  del "%TEMP%\python-installer.exe" >nul 2>&1
  if not exist "%PY_LOCAL%" (echo Python install failed. & pause & exit /b 1)
  set PY="%PY_LOCAL%"
)
echo == Using:
%PY% --version

rem ---- 2. WebView2 (the window engine; built into Windows 11 and most Windows 10) ----
set WV2=
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set WV2=1
reg query "HKCU\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set WV2=1
if not defined WV2 (
  echo == Installing the Microsoft WebView2 runtime
  powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' -OutFile \"$env:TEMP\wv2.exe\""
  "%TEMP%\wv2.exe" /silent /install
  del "%TEMP%\wv2.exe" >nul 2>&1
)

rem ---- 3. packages, in a Windows venv of its own --------------------------
if not exist .build-venv\Scripts\activate.bat (
  if exist .build-venv rmdir /s /q .build-venv
  %PY% -m venv .build-venv
)
call .build-venv\Scripts\activate.bat
echo == Installing packages (first time: a few minutes)
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
if errorlevel 1 (echo PACKAGE INSTALL FAILED & pause & exit /b 1)

rem ---- 4. build -------------------------------------------------------------
echo == Building OilStudio
pyinstaller --noconfirm --clean --workpath .pyi-work OilStudio.spec
if errorlevel 1 (echo BUILD FAILED & pause & exit /b 1)

powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\OilStudio' -DestinationPath 'dist\OilStudio-windows.zip'"
echo.
echo Done:  dist\OilStudio\OilStudio.exe
echo Share: dist\OilStudio-windows.zip  (the whole folder - the .exe alone will not run)
start "" "dist\OilStudio"
pause
