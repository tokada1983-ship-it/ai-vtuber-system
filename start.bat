@echo off
rem ============================================================
rem  AI Vtuber launcher
rem  NOTE: keep this file ASCII only. Japanese text in a UTF-8 batch
rem  file makes cmd.exe misread line offsets after "chcp 65001"
rem  (lines get skipped and the window closes right away).
rem ============================================================
chcp 65001 > nul
title AI Vtuber
cd /d "%~dp0"

echo.
echo  ========================================
echo    AI Vtuber kido chu...
echo  ========================================
echo.

rem -- find a Python that has the dependencies ---------------------
set "PYCMD="
if exist ".venv\Scripts\python.exe" call :TRY_PY ".venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" call :TRY_PY "venv\Scripts\python.exe"
call :TRY_PY py -3.10
call :TRY_PY py -3.11
call :TRY_PY py -3.12
call :TRY_PY py
call :TRY_PY python
if not defined PYCMD goto NO_DEPS
echo  [INFO] Python: %PYCMD%

rem -- read the port from config.yaml (server.port), default 9000 ---
set "PORT=9000"
for /f "usebackq delims=" %%P in (`%PYCMD% -c "from backend.config import load_config; print(load_config().server.port)" 2^>nul`) do set "PORT=%%P"
echo  [INFO] port: %PORT%

rem -- already running? --------------------------------------------
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" > nul 2>&1
if %errorlevel% == 0 (
    echo  [INFO] sudeni kido shiteimasu. browser wo hirakimasu.
    goto OPEN_BROWSER
)

echo  [INFO] server wo kido shiteimasu...
start "AI Vtuber Server" /min cmd /c "%PYCMD% run.py & pause"

set /a COUNT=0
:WAIT_LOOP
rem timeout exits at once when stdin is redirected, so wait with ping
ping -n 2 127.0.0.1 > nul 2>&1
set /a COUNT+=1
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" > nul 2>&1
if %errorlevel% == 0 goto SERVER_READY
if %COUNT% geq 60 goto TIMEOUT
echo  [INFO] kido machi... (%COUNT% byou)
goto WAIT_LOOP

:TIMEOUT
echo.
echo  [ERROR] server no kido ga timeout shimashita.
echo  "AI Vtuber Server" window no error wo kakunin shite kudasai.
echo.
pause
exit /b 1

:NO_DEPS
echo.
echo  [ERROR] fastapi / uvicorn ga haitta Python ga mitsukarimasen.
echo.
echo   install shite kudasai:
echo     py -3.10 -m pip install -r requirements.txt
echo.
pause
exit /b 1

:SERVER_READY
echo  [INFO] server kido kanryo!

:OPEN_BROWSER
start "" http://127.0.0.1:%PORT%

echo.
echo  ========================================
echo    kido kanryo!
echo.
echo    Dashboard   : http://127.0.0.1:%PORT%
echo    Chat        : http://127.0.0.1:%PORT%/chat
echo    Persona     : http://127.0.0.1:%PORT%/persona
echo    Avatar(OBS) : http://127.0.0.1:%PORT%/avatar
echo    Subtitle    : http://127.0.0.1:%PORT%/subtitle
echo.
echo    Teishi suru niwa "AI Vtuber Server" window wo tojite kudasai.
echo  ========================================
echo.
pause
exit /b 0

rem -- subroutine: use the first Python that can import the deps ----
:TRY_PY
if defined PYCMD goto :eof
%* -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% == 0 set "PYCMD=%*"
goto :eof