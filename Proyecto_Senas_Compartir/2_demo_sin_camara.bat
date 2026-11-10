@echo off
cd /d "%~dp0"
echo Iniciando la demo con el video de ejemplo y la Pico simulada...
echo Se abrira el navegador en unos segundos. Cierra esta ventana para apagar.
start "" cmd /c "timeout /t 10 >nul & start http://localhost:5000"
python servidor_web.py --camara demo_val.mp4 --simular
pause
