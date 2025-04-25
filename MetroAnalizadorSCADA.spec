# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['c:\\Users\\hans.cifuentes\\Desktop\\Python\\analizador_confiabilidad\\analizador_confiabilidad2\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('c:\\Users\\hans.cifuentes\\Desktop\\Python\\analizador_confiabilidad\\analizador_confiabilidad2/gui', 'gui'), ('c:\\Users\\hans.cifuentes\\Desktop\\Python\\analizador_confiabilidad\\analizador_confiabilidad2/processors', 'processors'), ('c:\\Users\\hans.cifuentes\\Desktop\\Python\\analizador_confiabilidad\\analizador_confiabilidad2/dashboard', 'dashboard')],
    hiddenimports=['ttkthemes', 'sklearn.utils._sorting', 'sklearn.metrics.pairwise', 'plotly.graph_objs', 'plotly.express', 'dash_bootstrap_components', 'statsmodels.tsa.arima.model', 'narwhals', 'dash.dependencies', 'dash.development', 'dash_bootstrap_components'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MetroAnalizadorSCADA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['c:\\Users\\hans.cifuentes\\Desktop\\Python\\analizador_confiabilidad\\analizador_confiabilidad2\\icons\\logo_metro.ico'],
)
