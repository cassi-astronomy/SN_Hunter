@echo off
cd /d "%~dp0"
python -m sn_hunter.app
if errorlevel 1 (
  echo.
  echo SN Hunter se nepodarilo spustit. Zkontrolujte instalaci podle README.md.
  pause
)
