@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem conda 환경 'fall'이 있으면 그 python을, 없으면 PATH의 python을 쓴다
set "PY=python"
if exist "%USERPROFILE%\miniconda3\envs\fall\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\fall\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\fall\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\fall\python.exe"
echo 낙상 감시 모니터를 켭니다. 브라우저가 자동으로 열립니다. 끄려면 이 창에서 Ctrl+C
"%PY%" -X utf8 monitor\server.py
pause
