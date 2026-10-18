@echo off
setlocal
rem Builds the PROTECTED OilStudio for clients: no readable code inside.
rem   dist-protected\OilStudio\OilStudio.exe  and  dist-protected\OilStudio-windows.zip
rem The Python code is compiled to machine code (Cython), the scripts and
rem templates are minified and packed inside the program. See build\protect.py.
rem Needs (installed automatically when possible): Python 3.10+, the free
rem "Build Tools for Visual Studio" (C++). Node.js is optional (scrambles the
rem JavaScript names too).
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
  "%TEMP%\python-installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0 Include_test=0
  del "%TEMP%\python-installer.exe" >nul 2>&1
  if not exist "%PY_LOCAL%" (echo Python install failed. & pause & exit /b 1)
  set PY="%PY_LOCAL%"
)
echo == Using:
%PY% --version

rem ---- 2. the C++ compiler (Build Tools for Visual Studio) ---------------
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
set HAVE_CL=
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set HAVE_CL=1
)
if not defined HAVE_CL (
  echo == The C++ compiler is missing: installing "Build Tools for Visual Studio"
  echo    ^(free, about 2-4 GB, 10-20 minutes; Windows may ask for permission^)
  winget --version >nul 2>&1
  if errorlevel 1 (
    echo winget is not available. Install it by hand, then run this again:
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo   - choose the workload "Desktop development with C++"
    pause & exit /b 1
  )
  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  if errorlevel 1 (
    echo The automatic install did not finish. Install it by hand, then run this again:
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo   - choose the workload "Desktop development with C++"
    pause & exit /b 1
  )
)

rem ---- 3. WebView2 (the window engine) ------------------------------------
set WV2=
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set WV2=1
reg query "HKCU\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>&1 && set WV2=1
if not defined WV2 (
  echo == Installing the Microsoft WebView2 runtime
  powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' -OutFile \"$env:TEMP\wv2.exe\""
  "%TEMP%\wv2.exe" /silent /install
  del "%TEMP%\wv2.exe" >nul 2>&1
)

rem ---- 4. packages -----------------------------------------------------------
if not exist .build-venv\Scripts\activate.bat (
  if exist .build-venv rmdir /s /q .build-venv
  %PY% -m venv .build-venv
)
call .build-venv\Scripts\activate.bat
echo == Installing packages
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller cython setuptools rjsmin rcssmin
if errorlevel 1 (echo PACKAGE INSTALL FAILED & pause & exit /b 1)

rem ---- 5. protect: minify, pack, compile, test --------------------------------
where node >nul 2>&1 || echo == Tip: install Node.js from nodejs.org to scramble the JavaScript names too
python build\protect.py --out .release
if errorlevel 1 (echo PROTECTING FAILED - see the message above & pause & exit /b 1)

rem ---- 6. build the app from the protected copy --------------------------------
echo == Building OilStudio (protected)
pushd .release
pyinstaller --noconfirm --clean --workpath ..\.pyi-work-protected --distpath ..\dist-protected OilStudio-protected.spec
set RC=%errorlevel%
popd
if not "%RC%"=="0" (echo BUILD FAILED & pause & exit /b 1)

copy /y build\CLIENT-README.txt "dist-protected\READ ME.txt" >nul
powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist-protected\OilStudio','dist-protected\READ ME.txt' -DestinationPath 'dist-protected\OilStudio-windows.zip'"
echo.
echo Done:  dist-protected\OilStudio\OilStudio.exe
echo Send:  dist-protected\OilStudio-windows.zip   (no readable code inside)
start "" "dist-protected"
pause
