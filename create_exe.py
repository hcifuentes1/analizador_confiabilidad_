import os
import sys
import PyInstaller.__main__

def create_executable():
    # Ruta base del proyecto
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Ruta del script principal
    main_script = os.path.join(base_dir, 'main.py')
    
    # Ruta del ícono
    icon_path = os.path.join(base_dir, 'icons', 'logo_metro.ico')
    
    # Opciones de PyInstaller
    pyinstaller_args = [
        '--onefile',           # Un solo archivo ejecutable
        '--windowed',          # Sin consola
        '--name', 'MetroAnalizadorSCADA',
        '--icon', icon_path,
        '--add-data', f'{base_dir}/gui:gui',
        '--add-data', f'{base_dir}/processors:processors',
        '--add-data', f'{base_dir}/dashboard:dashboard',
        '--hidden-import', 'ttkthemes',
        '--hidden-import', 'sklearn.utils._sorting',
        '--hidden-import', 'sklearn.metrics.pairwise',
        '--hidden-import', 'plotly.graph_objs',
        '--hidden-import', 'plotly.express',
        '--hidden-import', 'dash_bootstrap_components',
        '--hidden-import', 'statsmodels.tsa.arima.model',
        '--hidden-import', 'narwhals',
        '--hidden-import', 'dash.dependencies',
        '--hidden-import', 'dash.development',
        '--hidden-import', 'dash_bootstrap_components',
        main_script
    ]
    
    # Ejecutar PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)

if __name__ == '__main__':
    create_executable()