# processors/velcom_processor.py
import os
import pandas as pd
import datetime
import re
# Añadir al inicio del archivo
from dashboard.velcom_dashboard import create_ml_analysis_tab

class VelcomProcessor:
    """Clase para procesar archivos de Velocidad Comercial (Velcom)"""
    
    def __init__(self):
        self.file_path = None
        self.output_path = None
        self.data = None
        self.trains_data = None
        self.stations_data = None
        self.progress_callback = None
        
    def set_paths(self, file_path, output_path):
        """Establecer rutas de origen y destino"""
        self.file_path = file_path
        self.output_path = output_path
        
    def set_progress_callback(self, callback):
        """Establecer callback para actualizar progreso"""
        self.progress_callback = callback
        
    def update_progress(self, progress, message):
        """Actualizar progreso del procesamiento"""
        if self.progress_callback:
            self.progress_callback(progress, message)
            

    
    def process_file(self):
        """Procesar archivo Velcom y extraer datos"""
        if not self.file_path or not os.path.exists(self.file_path):
            self.update_progress(0, "Error: Archivo no encontrado")
            return False
            
        try:
            self.update_progress(10, "Leyendo archivo Velcom...")
            
            # Intentar diferentes codificaciones
            content = None
            encodings_to_try = ['latin1', 'cp1252', 'iso-8859-1', 'utf-8']
            
            for encoding in encodings_to_try:
                try:
                    with open(self.file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    self.update_progress(15, f"Archivo leído con codificación {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                self.update_progress(0, "Error: No se pudo leer el archivo con ninguna codificación compatible")
                return False
            
            # Extraer fechas de inicio y fin del reporte
            start_date_match = re.search(r'Inicio: \s*,\s*,\s*([0-9-]+ [0-9:]+)', content)
            end_date_match = re.search(r'Fin : \s*,\s*,\s*([0-9-]+ [0-9:]+)', content)
            
            start_date = start_date_match.group(1) if start_date_match else None
            end_date = end_date_match.group(1) if end_date_match else None
            
            self.update_progress(20, "Extrayendo datos de trenes...")
            
            # Extraer datos de trenes
            train_data_pattern = r',(\d+),([^,]+),(\d+),([^,]+),,([^,]*),,([^,]*)'
            train_data_matches = re.findall(train_data_pattern, content)
            
            records = []
            for match in train_data_matches:
                # Verificar que no sea una línea de encabezado
                if len(match) >= 6 and match[0].isdigit():
                    train_number = match[0]
                    material = match[1]
                    track = match[2]
                    station = match[3]
                    arrival_time = match[4].strip() if match[4].strip() else None
                    departure_time = match[5].strip() if match[5].strip() else None
                    
                    records.append({
                        'train_number': train_number,
                        'material': material,
                        'track': track,
                        'station': station,
                        'arrival_time': arrival_time,
                        'departure_time': departure_time
                    })
            
            # Convertir a DataFrame
            self.data = pd.DataFrame(records)
            
            self.update_progress(50, f"Datos extraídos: {len(self.data)} registros de movimientos")
            
            # Convertir columnas de tiempo a datetime
            if not self.data.empty:
                self.data['arrival_time'] = pd.to_datetime(self.data['arrival_time'], format="%d/%m/%Y  %H:%M:%S", errors='coerce')
                self.data['departure_time'] = pd.to_datetime(self.data['departure_time'], format="%d/%m/%Y  %H:%M:%S", errors='coerce')
            
            # Obtener estadísticas por tren
            self.update_progress(70, "Generando estadísticas por tren...")
            self.trains_data = self.data.groupby(['train_number', 'material']).agg({
                'station': 'count',
                'arrival_time': ['min', 'max']
            }).reset_index()
            
            self.trains_data.columns = [
                'train_number', 'material', 'stations_count', 
                'first_arrival', 'last_arrival'
            ]
            
            # Obtener estadísticas por estación
            self.update_progress(80, "Generando estadísticas por estación...")
            self.stations_data = self.data.groupby(['station']).agg({
                'train_number': 'count',
                'track': lambda x: x.astype(int).mean(),
                'arrival_time': 'count'
            }).reset_index()
            
            self.stations_data.columns = [
                'station', 'train_count', 'avg_track', 'arrival_count'
            ]
            
            # Guardar datos procesados
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
                
            self.update_progress(90, "Guardando datos procesados...")
            
            # Guardar datos en archivos CSV
            self.data.to_csv(os.path.join(self.output_path, 'velcom_data.csv'), index=False)
            self.trains_data.to_csv(os.path.join(self.output_path, 'velcom_trains.csv'), index=False)
            self.stations_data.to_csv(os.path.join(self.output_path, 'velcom_stations.csv'), index=False)
            
            # Guardar fechas de inicio y fin
            report_info = {
                'start_date': start_date,
                'end_date': end_date,
                'records_count': len(self.data),
                'trains_count': len(self.data['train_number'].unique()),
                'stations_count': len(self.data['station'].unique()),
                'processing_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            pd.DataFrame([report_info]).to_csv(os.path.join(self.output_path, 'velcom_info.csv'), index=False)
            
            self.update_progress(100, "Procesamiento completado")
            return True
            
        except Exception as e:
            self.update_progress(0, f"Error al procesar archivo: {str(e)}")
            return False
    
    def get_train_info(self, train_number=None):
        """Obtener información detallada de un tren específico"""
        if self.data is None:
            return None
            
        if train_number:
            return self.data[self.data['train_number'] == train_number].sort_values('arrival_time')
        else:
            return self.trains_data.sort_values('train_number')
    
    def get_station_info(self, station=None):
        """Obtener información detallada de una estación específica"""
        if self.data is None:
            return None
            
        if station:
            return self.data[self.data['station'] == station].sort_values('arrival_time')
        else:
            return self.stations_data.sort_values('station')
    
    def get_time_range_info(self, start_time, end_time):
        """Obtener información de trenes en un rango de tiempo específico"""
        if self.data is None:
            return None
            
        mask = (self.data['arrival_time'] >= start_time) & (self.data['arrival_time'] <= end_time)
        return self.data[mask].sort_values('arrival_time')