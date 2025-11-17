@echo off
echo Starting Backend Server...
echo.
cd backend
call venv\Scripts\activate
python -m app.main
