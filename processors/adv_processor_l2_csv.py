# processors/adv_processor_l2_csv.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class ADVProcessorL2CSV(BaseProcessor):
    """Procesador para datos ADV (Aparatos de Vía) de la Línea 2 en formato CSV"""
    
    def __init__(self):
        super().__init__(line="L2", analysis_type="ADV")
        # Atributos específicos para ADV L2
        self.df = None
        self.df_L2_ADV_DISC = None  # Discordancias de agujas
        self.df_L2_ADV_MOV = None   # Movimientos de agujas
        self.df_L2_ADV_TIME = None  # Tiempos de movimiento
        
        # Atributo para el tipo de datos
        self.data_type = "Sacem"  # Valor por defecto
        
        # Umbral para detección de tiempos anómalos (porcentaje sobre la media)
        self.time_threshold = 1.5
    
    def set_data_type(self, data_type):
        """Establecer tipo de datos (Sacem o SCADA)"""
        self.data_type = data_type
    
    def find_files(self):
        """Encontrar archivos CSV para análisis ADV de Línea 2"""
        self.csv_files = []
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Recorrer carpetas y encontrar archivos CSV
        for root, dirs, files in os.walk(self.root_folder_path):
            for file in files:
                # Buscar archivos con patrones que indiquen datos de agujas
                if file.lower().endswith('.csv') and any(pattern in file for pattern in ["kag", "adv", "aguja"]):
                    self.csv_files.append(os.path.join(root, file))
        
        return len(self.csv_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos CSV para análisis ADV de Línea 2"""
        movements_data = []  # Para almacenar datos de movimientos
        discordance_data = []  # Para almacenar datos de discordancias
        total_files = len(self.csv_files)
        
        for i, file_path in enumerate(self.csv_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(file_path)}")
                
                # Intentar determinar el separador correcto
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                
                # Determinar el separador basándose en la primera línea
                if ',' in first_line:
                    separator = ','
                elif ';' in first_line:
                    separator = ';'
                else:
                    # Si no podemos determinar el separador, probamos con coma por defecto
                    separator = ','
                
                # Leer el archivo CSV
                try:
                    # Para versiones recientes de pandas (1.3.0+)
                    df = pd.read_csv(file_path, sep=separator, encoding='utf-8', on_bad_lines='warn')
                except TypeError:
                    # Para versiones anteriores de pandas
                    df = pd.read_csv(file_path, sep=separator, encoding='utf-8', error_bad_lines=False)
                
                # Verificar si el archivo tiene las columnas esperadas
                if 'ciclo' in df.columns and 'tiempo' in df.columns:
                    # Este es el formato con columnas "ciclo" y "tiempo"
                    df['Fecha Hora'] = pd.to_datetime(df['tiempo'], errors='coerce')
                    
                    # Identificar columnas de aparatos de vía (Kag)
                    kag_columns = [col for col in df.columns if 'Kag' in col]
                    
                    if kag_columns:
                        # Procesar pares de aparatos de vía (izquierda/derecha)
                        kag_pairs = self._identify_kag_pairs(kag_columns)
                        
                        for kag_base, kag_left, kag_right in kag_pairs:
                            # Extraer datos de este aparato de vía
                            kag_data = self._process_kag(df, kag_base, kag_left, kag_right, progress_callback)
                            
                            if kag_data:
                                movements_data.extend(kag_data['movements'])
                                discordance_data.extend(kag_data['discordances'])
            
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error en archivo {file_path}: {str(e)}")
        
        # Crear DataFrames de movimientos y discordancias
        if movements_data:
            self.df_L2_ADV_MOV = pd.DataFrame(movements_data)
        
        if discordance_data:
            self.df_L2_ADV_DISC = pd.DataFrame(discordance_data)
        
        # Preparar DataFrame principal para análisis posterior
        if self.df_L2_ADV_MOV is not None:
            self.df = self.df_L2_ADV_MOV.copy()
            return True
        else:
            return False
    
    def _identify_kag_pairs(self, kag_columns):
        """Identificar pares de aparatos de vía (izquierda/derecha)"""
        kag_pairs = []
        
        # Extraer los nombres base de los aparatos de vía
        kag_bases = set()
        for col in kag_columns:
            # Extraer el nombre base (sin la posición izquierda/derecha)
            parts = col.split()
            if len(parts) >= 2:
                kag_base = ' '.join(parts[:-1])  # Todo menos la última parte
                kag_bases.add(kag_base)
        
        # Para cada nombre base, encontrar sus posiciones izquierda/derecha
        for kag_base in kag_bases:
            kag_left = next((col for col in kag_columns if col.startswith(kag_base) and col.endswith("I")), None)
            kag_right = next((col for col in kag_columns if col.startswith(kag_base) and col.endswith("D")), None)
            
            if kag_left and kag_right:
                kag_pairs.append((kag_base, kag_left, kag_right))
        
        return kag_pairs
    
    def _process_kag(self, df, kag_base, kag_left, kag_right, progress_callback=None):
        """Procesar datos de un aparato de vía específico"""
        if progress_callback:
            progress_callback(None, f"Procesando aparato de vía: {kag_base}")
        
        # Extraer estación del nombre del aparato de vía
        station = self._extract_station_from_kag(kag_base)
        
        # Crear una copia de las columnas relevantes
        kag_df = df[['Fecha Hora', kag_left, kag_right]].copy()
        
        # Convertir a valores numéricos (para asegurar que sean 0 o 1)
        kag_df[kag_left] = pd.to_numeric(kag_df[kag_left], errors='coerce').fillna(0).astype(int)
        kag_df[kag_right] = pd.to_numeric(kag_df[kag_right], errors='coerce').fillna(0).astype(int)
        
        # Ordenar por fecha y hora
        kag_df = kag_df.sort_values('Fecha Hora')
        
        # Crear columna de estado combinado (para detectar movimientos y discordancias)
        kag_df['estado_combinado'] = kag_df[kag_left].astype(str) + kag_df[kag_right].astype(str)
        
        # Detectar cambios de estado
        kag_df['estado_previo'] = kag_df['estado_combinado'].shift(1)
        kag_df['cambio_estado'] = (kag_df['estado_combinado'] != kag_df['estado_previo']).astype(int)
        
        # Preparar para detectar movimientos (ambos en 0)
        kag_df['en_movimiento'] = ((kag_df[kag_left] == 0) & (kag_df[kag_right] == 0)).astype(int)
        
        # Detectar inicio y fin de movimientos
        kag_df['inicio_movimiento'] = (kag_df['en_movimiento'] == 1) & (kag_df['en_movimiento'].shift(1) == 0)
        kag_df['fin_movimiento'] = (kag_df['en_movimiento'] == 0) & (kag_df['en_movimiento'].shift(1) == 1)
        
        # Detectar discordancias (ambos en 1 o transiciones inválidas)
        kag_df['discordancia'] = (
            (kag_df[kag_left] == 1) & (kag_df[kag_right] == 1) |  # Ambos en 1 (inválido)
            ((kag_df['estado_combinado'] == '10') & (kag_df['estado_previo'] == '01') & ~kag_df['inicio_movimiento']) |  # Transición directa sin movimiento
            ((kag_df['estado_combinado'] == '01') & (kag_df['estado_previo'] == '10') & ~kag_df['inicio_movimiento'])     # Transición directa sin movimiento
        )
        
        # Recolectar datos de movimientos
        movements = []
        current_movement_start = None
        
        # Recolectar datos de discordancias
        discordances = []
        
        for idx, row in kag_df.iterrows():
            # Registrar inicio de movimiento
            if row['inicio_movimiento']:
                current_movement_start = row['Fecha Hora']
            
            # Registrar fin de movimiento y calcular duración
            if row['fin_movimiento'] and current_movement_start is not None:
                movement_duration = (row['Fecha Hora'] - current_movement_start).total_seconds()
                
                movements.append({
                    'Fecha': row['Fecha Hora'].date(),
                    'Hora Inicio': current_movement_start.time(),
                    'Hora Fin': row['Fecha Hora'].time(),
                    'Duración (s)': movement_duration,
                    'Equipo': kag_base,
                    'Estación': station,
                    'Estado Anterior': row['estado_previo'],
                    'Estado Nuevo': row['estado_combinado']
                })
                
                current_movement_start = None
            
            # Registrar discordancias
            if row['discordancia']:
                discordances.append({
                    'Fecha Hora': row['Fecha Hora'],
                    'Equipo': kag_base,
                    'Estación': station,
                    'Estado Izquierda': row[kag_left],
                    'Estado Derecha': row[kag_right],
                    'Estado Combinado': row['estado_combinado'],
                    'Tipo': 'Ambos lados activos' if row[kag_left] == 1 and row[kag_right] == 1 else 'Transición directa'
                })
        
        return {
            'movements': movements,
            'discordances': discordances
        }
    
    def _extract_station_from_kag(self, kag_name):
        """Extraer información de estación del nombre del aparato de vía"""
        # Intentar extraer la estación del nombre (última parte generalmente)
        parts = kag_name.split()
        if len(parts) >= 3:
            return parts[-1]  # La última parte suele ser la estación
        
        # Si no se puede determinar, devolver desconocido
        return 'Desconocido'
    
    def preprocess_data(self, progress_callback=None):
        """Realizar análisis estadístico de tiempos de movimiento"""
        if progress_callback:
            progress_callback(25, "Analizando tiempos de movimiento...")
        
        if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
            # Asegurar que Duración es numérica
            self.df_L2_ADV_MOV['Duración (s)'] = pd.to_numeric(self.df_L2_ADV_MOV['Duración (s)'], errors='coerce')
            
            # Calcular estadísticas de tiempo por equipo
            self.df_L2_ADV_TIME = self.df_L2_ADV_MOV.groupby('Equipo')['Duración (s)'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).reset_index()
            
            # Calcular umbrales para detección de anomalías
            self.df_L2_ADV_TIME['umbral_superior'] = self.df_L2_ADV_TIME['mean'] + (self.df_L2_ADV_TIME['std'] * self.time_threshold)
            
            # Identificar movimientos anormalmente largos
            self.df_L2_ADV_MOV = self.df_L2_ADV_MOV.merge(
                self.df_L2_ADV_TIME[['Equipo', 'mean', 'umbral_superior']], 
                on='Equipo',
                how='left'
            )
            
            # Marcar los movimientos anómalos
            self.df_L2_ADV_MOV['Anomalía'] = self.df_L2_ADV_MOV['Duración (s)'] > self.df_L2_ADV_MOV['umbral_superior']
            
            # Crear ID único para cada registro
            self.df_L2_ADV_MOV['Fecha'] = pd.to_datetime(self.df_L2_ADV_MOV['Fecha']).dt.strftime('%Y-%m-%d')
            self.df_L2_ADV_MOV['ID_Movimiento'] = (
                self.df_L2_ADV_MOV['Fecha'] + '_' + 
                self.df_L2_ADV_MOV['Equipo'] + '_' + 
                self.df_L2_ADV_MOV['Hora Inicio'].astype(str)
            )
            
            # Agregar columna de línea
            self.df_L2_ADV_MOV['Linea'] = 'L2'
        
        if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
            # Crear ID único para cada discordancia
            self.df_L2_ADV_DISC['Fecha Hora'] = pd.to_datetime(self.df_L2_ADV_DISC['Fecha Hora']).dt.strftime('%Y-%m-%d %H:%M:%S')
            self.df_L2_ADV_DISC['ID_Discordancia'] = (
                self.df_L2_ADV_DISC['Fecha Hora'] + '_' + 
                self.df_L2_ADV_DISC['Equipo']
            )
            
            # Agregar columna de línea
            self.df_L2_ADV_DISC['Linea'] = 'L2'
        
        if progress_callback:
            progress_callback(40, "Preprocesamiento completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """Detectar aparatos de vía con comportamiento anómalo"""
        if progress_callback:
            progress_callback(70, "Detectando agujas con comportamiento anómalo...")
        
        if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
            # Contar anomalías por equipo
            anomalies_count = self.df_L2_ADV_MOV.groupby('Equipo')['Anomalía'].sum().reset_index()
            anomalies_count.columns = ['Equipo', 'Total_Anomalías']
            
            # Calcular porcentaje de anomalías
            anomalies_percent = self.df_L2_ADV_MOV.groupby('Equipo').agg(
                Total_Movimientos=('Anomalía', 'count'),
                Total_Anomalías=('Anomalía', 'sum')
            ).reset_index()
            
            anomalies_percent['Porcentaje_Anomalías'] = (
                100 * anomalies_percent['Total_Anomalías'] / anomalies_percent['Total_Movimientos']
            )
            
            # Identificar equipos con alta frecuencia de anomalías (>10%)
            critical_equipments = anomalies_percent[anomalies_percent['Porcentaje_Anomalías'] > 10]
            
            # Guardar datos para reportes
            self.critical_equipments = critical_equipments
        
        if progress_callback:
            progress_callback(80, "Detección de anomalías completada")
        
        return True
    
    def prepare_reports(self, progress_callback=None):
        """Preparar los diferentes reportes"""
        if progress_callback:
            progress_callback(85, "Preparando reportes...")
        
        # Agregar información adicional a los DataFrames existentes
        if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
            # Convertir a formato adecuado para reportes
            self.df_L2_ADV_MOV['Hora Inicio'] = self.df_L2_ADV_MOV['Hora Inicio'].astype(str)
            self.df_L2_ADV_MOV['Hora Fin'] = self.df_L2_ADV_MOV['Hora Fin'].astype(str)
            
            # Eliminar columnas de análisis estadístico del reporte final
            if 'mean' in self.df_L2_ADV_MOV.columns:
                self.df_L2_ADV_MOV = self.df_L2_ADV_MOV.drop(['mean', 'umbral_superior'], axis=1)
            
            # Crear reporte de movimientos agrupados por día
            self.df_L2_ADV_MOV_DAILY = self.df_L2_ADV_MOV.groupby(['Fecha', 'Equipo', 'Estación']).agg(
                Total_Movimientos=('ID_Movimiento', 'count'),
                Promedio_Duración=('Duración (s)', 'mean'),
                Total_Anomalías=('Anomalía', 'sum')
            ).reset_index()
            
            # Crear ID único para cada registro diario
            self.df_L2_ADV_MOV_DAILY['ID'] = (
                self.df_L2_ADV_MOV_DAILY['Fecha'] + '_' + 
                self.df_L2_ADV_MOV_DAILY['Equipo']
            )
            
            # Agregar columna de línea
            self.df_L2_ADV_MOV_DAILY['Linea'] = 'L2'
        
        if progress_callback:
            progress_callback(90, "Reportes preparados")
        
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(90, "Actualizando reportes existentes...")
            
            # 1. Actualizar reporte de discordancias
            if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC_Mensual.csv')
                
                if os.path.exists(disc_file_path):
                    # Cargar reporte existente
                    df_L2_ADV_DISC_Mensual = pd.read_csv(disc_file_path)
                    
                    # Concatenar con nuevos datos
                    df_L2_ADV_DISC_Mensual = pd.concat([df_L2_ADV_DISC_Mensual, self.df_L2_ADV_DISC], ignore_index=True)
                    
                    # Eliminar duplicados
                    if 'ID_Discordancia' in df_L2_ADV_DISC_Mensual.columns:
                        df_L2_ADV_DISC_Mensual.drop_duplicates(subset=['ID_Discordancia'], inplace=True)
                    
                    # Guardar reporte actualizado
                    df_L2_ADV_DISC_Mensual.to_csv(disc_file_path, index=False)
                else:
                    # Crear nuevo reporte
                    self.df_L2_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # 2. Actualizar reporte de movimientos
            if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV_Mensual.csv')
                
                if os.path.exists(mov_file_path):
                    # Cargar reporte existente
                    df_L2_ADV_MOV_Mensual = pd.read_csv(mov_file_path)
                    
                    # Concatenar con nuevos datos
                    df_L2_ADV_MOV_Mensual = pd.concat([df_L2_ADV_MOV_Mensual, self.df_L2_ADV_MOV], ignore_index=True)
                    
                    # Eliminar duplicados
                    if 'ID_Movimiento' in df_L2_ADV_MOV_Mensual.columns:
                        df_L2_ADV_MOV_Mensual.drop_duplicates(subset=['ID_Movimiento'], inplace=True)
                    
                    # Guardar reporte actualizado
                    df_L2_ADV_MOV_Mensual.to_csv(mov_file_path, index=False)
                else:
                    # Crear nuevo reporte
                    self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
            
            # 3. Actualizar reporte diario de movimientos
            if hasattr(self, 'df_L2_ADV_MOV_DAILY') and self.df_L2_ADV_MOV_DAILY is not None and not self.df_L2_ADV_MOV_DAILY.empty:
                daily_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV_DAILY_Mensual.csv')
                
                if os.path.exists(daily_file_path):
                    # Cargar reporte existente
                    df_L2_ADV_MOV_DAILY_Mensual = pd.read_csv(daily_file_path)
                    
                    # Concatenar con nuevos datos
                    df_L2_ADV_MOV_DAILY_Mensual = pd.concat([df_L2_ADV_MOV_DAILY_Mensual, self.df_L2_ADV_MOV_DAILY], ignore_index=True)
                    
                    # Eliminar duplicados
                    if 'ID' in df_L2_ADV_MOV_DAILY_Mensual.columns:
                        df_L2_ADV_MOV_DAILY_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Guardar reporte actualizado
                    df_L2_ADV_MOV_DAILY_Mensual.to_csv(daily_file_path, index=False)
                else:
                    # Crear nuevo reporte
                    self.df_L2_ADV_MOV_DAILY.to_csv(daily_file_path, index=False)
            
            if progress_callback:
                progress_callback(95, "Reportes actualizados correctamente")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al actualizar reportes: {str(e)}")
            return False
    
    def save_dataframe(self):
        """Guardar los DataFrames principales"""
        try:
            # Guardar DataFrame de movimientos
            if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
                self.df_L2_ADV_MOV.to_csv(os.path.join(self.output_folder_path, 'df_L2_ADV_MOV.csv'), index=False)
            
            # Guardar DataFrame de discordancias
            if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
                self.df_L2_ADV_DISC.to_csv(os.path.join(self.output_folder_path, 'df_L2_ADV_DISC.csv'), index=False)
            
            # Guardar DataFrame de estadísticas de tiempo
            if self.df_L2_ADV_TIME is not None and not self.df_L2_ADV_TIME.empty:
                self.df_L2_ADV_TIME.to_csv(os.path.join(self.output_folder_path, 'df_L2_ADV_TIME.csv'), index=False)
            
            # Guardar DataFrame de movimientos diarios
            if hasattr(self, 'df_L2_ADV_MOV_DAILY') and self.df_L2_ADV_MOV_DAILY is not None and not self.df_L2_ADV_MOV_DAILY.empty:
                self.df_L2_ADV_MOV_DAILY.to_csv(os.path.join(self.output_folder_path, 'df_L2_ADV_MOV_DAILY.csv'), index=False)
            
            return True
        except Exception as e:
            print(f"Error al guardar DataFrames: {str(e)}")
            return False