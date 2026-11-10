@echo off
cd /d "%~dp0"
echo Instalando las librerias del proyecto (puede tardar unos minutos)...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: no se pudo instalar. Revisa que Python este instalado y en el PATH.
) else (
  echo.
  echo Listo. Ahora abre 2_demo_sin_camara.bat o 3_con_camara.bat
)
pause
