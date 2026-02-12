@echo off
cd /d "%~dp0"
py -3 -m pip install --upgrade pip && py -3 -m pip install pyyaml pandas numpy requests matplotlib openpyxl gspread google-auth oauth2client
py -3 gui_launcher.py
pause
