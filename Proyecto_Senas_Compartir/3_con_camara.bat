@echo off
cd /d "%~dp0"
echo Iniciando con la camara 0 (si tienes camara USB externa cambia --camara 0 por --camara 1).
echo Si la Raspberry Pi Pico esta conectada se usa sola; si no, queda en modo simulado.
start "" cmd /c "timeout /t 10 >nul & start http://localhost:5000"
python servidor_web.py --camara 0
pause
