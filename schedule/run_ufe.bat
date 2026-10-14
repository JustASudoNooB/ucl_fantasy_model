@echo off
REM Windows: run the engine's scheduler once (Task Scheduler calls this every 15 minutes).
REM Create the task once, from the project folder:
REM   schtasks /Create /SC MINUTE /MO 15 /TN "UCL Fantasy Engine" /TR "%CD%\schedule\run_ufe.bat"
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m ufe schedule --once >> data\processed\scheduler.log 2>&1
