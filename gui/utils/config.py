# utils/config.py
import os
import json
from datetime import datetime

class Config:
    """Gestionar la configuración de la aplicación"""
    
    def __init__(self):
        self.config_file = 'config.json'
        self.default_config = {
            'default_source_path': '',
            'default_output_path': '',
            'f_oc_1': 0.1,
            'f_lb_2': 0.05,
            'theme': 'arc',
            'recent_paths': []
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Cargar configuración desde archivo"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_config.copy()
        else:
            return self.default_config.copy()
    
    def save_config(self):
        """Guardar configuración a archivo"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except:
            return False
    
    def get(self, key, default=None):
        """Obtener valor de configuración"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Establecer valor de configuración"""
        self.config[key] = value
        self.save_config()
    
    def add_recent_path(self, path_type, path):
        """Añadir ruta reciente"""
        if 'recent_paths' not in self.config:
            self.config['recent_paths'] = []
        
        # Crear entrada para la ruta
        entry = {
            'type': path_type,
            'path': path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Reemplazado import_timestamp()
        }
        
        # Añadir al inicio y limitar a 10 entradas
        self.config['recent_paths'].insert(0, entry)
        self.config['recent_paths'] = self.config['recent_paths'][:10]
        
        self.save_config()
    
    def get_recent_paths(self, path_type=None):
        """Obtener rutas recientes, opcionalmente filtradas por tipo"""
        if 'recent_paths' not in self.config:
            return []
        
        if path_type:
            return [entry for entry in self.config['recent_paths'] if entry['type'] == path_type]
        else:
            return self.config['recent_paths']