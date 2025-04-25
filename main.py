# main.py
from gui.utils.main_window import MetroAnalyzerApp
from ttkthemes import ThemedTk

def main():
    """Función principal que inicia la aplicación"""
    root = ThemedTk(theme="scidgrey")
    app = MetroAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()