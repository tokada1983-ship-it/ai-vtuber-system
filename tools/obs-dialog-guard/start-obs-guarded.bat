@echo off
rem start-obs-guarded.bat
rem Starts OBS Studio (if not already running), then runs the crash-dialog
rem guard so a leftover crash dialog is dismissed in NORMAL mode and the
rem OBS WebSocket (:4455) comes up. Run in the interactive user session.
setlocal
set GUARD=<YOUR_PATH>\obs-dialog-guard\obs-dialog-guard.ps1
set OBS_EXE=
if exist "C:\Program Files\obs-studio\bin\64bit\obs64.exe" set OBS_EXE=C:\Program Files\obs-studio\bin\64bit\obs64.exe
if not defined OBS_EXE if exist "C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe" set OBS_EXE=C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe
if not defined OBS_EXE (
  echo [start-obs-guarded] obs64.exe not found.
  exit /b 2
)
tasklist /FI "IMAGENAME eq obs64.exe" 2>nul | find /I "obs64.exe" >nul
if errorlevel 1 (
  echo [start-obs-guarded] starting OBS: %OBS_EXE%
  pushd "%OBS_EXE%\..\"
  start "" "%OBS_EXE%"
  popd
) else (
  echo [start-obs-guarded] OBS already running; guard only.
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%GUARD%" -TimeoutSec 120
set RC=%ERRORLEVEL%
echo [start-obs-guarded] guard exit code: %RC%
exit /b %RC%
