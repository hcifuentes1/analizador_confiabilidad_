@echo off

REM Verificar si el entorno virtual está activado
if not defined VIRTUAL_ENV (
    echo Active el entorno virtual antes de ejecutar este script.
    echo Ejemplo: metro_venv\Scripts\activate
    pause
    exit /b
)

REM Crear ejecutable
python create_exe.py

REM Crear carpeta de distribución si no existe
if not exist dist mkdir dist

REM Mover ejecutable a carpeta de distribución
move MetroAnalizadorSCADA.exe dist\

echo Ejecutable creado con éxito.
pause