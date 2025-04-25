# processors/base_processor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class BaseProcessor:
    """Clase base para procesadores de datos del Metro de Santiago"""
    
    def __init__(self, line="L5", analysis_type="CDV"):
        self.line = line
        self.analysis_type = analysis_type
        self.root_folder_path = None
        self.output_folder_path = None
        self.txt_files = []
        self.df = None  # DataFrame principal
        
    def set_paths(self, root_folder_path, output_folder_path):
        """Establecer rutas de origen y destino"""
        self.root_folder_path = root_folder_path
        self.output_folder_path = output_folder_path
    
    def set_analysis_type(self, analysis_type):
        """Establecer tipo de análisis (CDV o ADV)"""
        self.analysis_type = analysis_type
    
    def find_files(self):
        """Método base para encontrar archivos - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def read_files(self, progress_callback=None):
        """Método base para leer archivos - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def preprocess_data(self, progress_callback=None):
        """Método base para preprocesar datos - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def detect_anomalies(self, progress_callback=None):
        """Método base para detectar anomalías - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def prepare_reports(self, progress_callback=None):
        """Método base para preparar reportes - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def update_reports(self, progress_callback=None):
        """Método base para actualizar reportes - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def save_dataframe(self):
        """Método base para guardar el DataFrame principal - debe ser implementado por las subclases"""
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def process_data(self, progress_callback=None):
        """Ejecutar el flujo completo de procesamiento de datos"""
        try:
            # 1. Encontrar archivos
            if progress_callback:
                progress_callback(0, f"Buscando archivos para análisis {self.analysis_type} en Línea {self.line}...")
            num_files = self.find_files()
            if num_files == 0:
                if progress_callback:
                    progress_callback(100, "No se encontraron archivos para procesar")
                return False
            
            # 2. Leer archivos
            if progress_callback:
                progress_callback(5, f"Leyendo {num_files} archivos...")
            if not self.read_files(progress_callback):
                return False
            
            # 3. Preprocesar datos
            if progress_callback:
                progress_callback(20, "Preprocesando datos...")
            self.preprocess_data(progress_callback)
            
            # 4. Detectar anomalías
            if progress_callback:
                progress_callback(70, f"Detectando anomalías para {self.analysis_type}...")
            self.detect_anomalies(progress_callback)
            
            # 5. Preparar reportes
            if progress_callback:
                progress_callback(80, "Preparando reportes...")
            self.prepare_reports(progress_callback)
            
            # 6. Actualizar reportes existentes
            if progress_callback:
                progress_callback(90, "Actualizando reportes existentes...")
            self.update_reports(progress_callback)
            
            # 7. Guardar DataFrame principal
            if progress_callback:
                progress_callback(95, "Guardando DataFrame principal...")
            self.save_dataframe()
            
            if progress_callback:
                progress_callback(100, f"Procesamiento {self.analysis_type} completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False