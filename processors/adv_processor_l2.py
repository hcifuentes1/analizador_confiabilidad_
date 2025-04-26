# processors/adv_processor_l2.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class ADVProcessorL2(BaseProcessor):
    """Procesador para datos ADV (Agujas) de la Línea 2"""
    
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
        """Encontrar archivos CSV/Excel para análisis ADV de Línea 2"""
        self.data_files = []
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Recorrer carpetas y encontrar archivos
        for root, dirs, files in os.walk(self.root_folder_path):
            for file in files:
                # Filtrar según el tipo de datos y formato del archivo
                if self.data_type == "Sacem":
                    # Buscar archivos relacionados con agujas
                    if file.endswith(('.xlsx', '.xls', '.csv')) and (
                        'AGS' in file or 'adv' in file.lower() or
                        'aguja' in file.lower() or 'kag' in file.lower()):
                        self.data_files.append(os.path.join(root, file))
                elif self.data_type == "SCADA":
                    # En el futuro, aquí se implementará la lógica para archivos SCADA
                    continue
        
        return len(self.data_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos para análisis ADV de Línea 2"""
        movements_data = []  # Para almacenar datos de movimientos
        discordance_data = []  # Para almacenar datos de discordancias
        total_files = len(self.data_files)
        
        for i, file_path in enumerate(self.data_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(file_path)}")
                
                # Determinar el tipo de archivo basado en su extensión y contenido
                if file_path.lower().endswith('.csv'):
                    # Verificar si es un archivo CSV con nuevo formato (Kag)
                    is_new_csv_format = self._check_if_new_csv_format(file_path)
                    
                    if is_new_csv_format:
                        # Procesar archivo CSV con nuevo formato
                        kag_data = self._process_new_csv_format(file_path, progress_callback)
                        if kag_data:
                            movements_data.extend(kag_data.get('movements', []))
                            discordance_data.extend(kag_data.get('discordances', []))
                    else:
                        # Procesar archivo CSV con formato antiguo
                        old_data = self._process_old_format_file(file_path, progress_callback)
                        if old_data:
                            movements_data.extend(old_data.get('movements', []))
                            discordance_data.extend(old_data.get('discordances', []))
                else:
                    # Procesar archivo Excel
                    excel_data = self._process_excel_file(file_path, progress_callback)
                    if excel_data:
                        movements_data.extend(excel_data.get('movements', []))
                        discordance_data.extend(excel_data.get('discordances', []))
            
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error en archivo {file_path}: {str(e)}")
        
        # Crear DataFrames de movimientos y discordancias
        if movements_data:
            self.df_L2_ADV_MOV = pd.DataFrame(movements_data)
            if progress_callback:
                progress_callback(None, f"Se procesaron {len(movements_data)} registros de movimientos")
        
        if discordance_data:
            self.df_L2_ADV_DISC = pd.DataFrame(discordance_data)
            if progress_callback:
                progress_callback(None, f"Se encontraron {len(discordance_data)} discordancias")
        
        # Resultado del procesamiento
        if self.df_L2_ADV_MOV is not None or self.df_L2_ADV_DISC is not None:
            return True
        else:
            if progress_callback:
                progress_callback(None, "No se encontraron datos de movimientos ni discordancias")
            return False
    
    def _check_if_new_csv_format(self, file_path):
        """Verificar si el archivo CSV está en el nuevo formato con columnas Kag"""
        try:
            # Leer las primeras líneas para verificar formato
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip()
            
            # Verificar si contiene columnas Kag
            return 'Kag' in header
        except:
            # Si hay error al leer, asumir que no es nuevo formato
            return False
    
    def _process_new_csv_format(self, file_path, progress_callback=None):
        """Procesar archivo CSV con nuevo formato (Kag)"""
        try:
            if progress_callback:
                progress_callback(None, f"Procesando archivo de nuevo formato: {os.path.basename(file_path)}")
            
            # Determinar el separador (coma o punto y coma)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            separator = ',' if ',' in first_line else ';'
            
            # Leer el CSV
            try:
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8', on_bad_lines='warn')
            except TypeError:
                # Para versiones anteriores de pandas
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8', error_bad_lines=False)
            
            # Verificar que existen las columnas necesarias
            if 'ciclo' not in df.columns or 'tiempo' not in df.columns:
                if progress_callback:
                    progress_callback(None, f"Archivo sin formato esperado: {os.path.basename(file_path)}")
                return None
            
            # Crear columna de fecha y hora
            df['Fecha Hora'] = pd.to_datetime(df['tiempo'], errors='coerce')
            
            # Identificar columnas de aparatos de vía (Kag)
            kag_columns = [col for col in df.columns if 'Kag' in col]
            
            if not kag_columns:
                if progress_callback:
                    progress_callback(None, f"No se encontraron columnas Kag en: {os.path.basename(file_path)}")
                return None
            
            # Procesar pares de aparatos de vía (izquierda/derecha)
            kag_pairs = self._identify_kag_pairs(kag_columns)
            
            movements = []
            discordances = []
            
            for kag_base, kag_left, kag_right in kag_pairs:
                # Extraer datos de este aparato de vía
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
                current_movement_start = None
                
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
            
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error procesando archivo nuevo formato: {str(e)}")
            return None
    
    def _identify_kag_pairs(self, kag_columns):
        """Identificar pares de aparatos de vía (izquierda/derecha)"""
        kag_pairs = []
        
        # Extraer los nombres base de los aparatos de vía
        kag_bases = set()
        for col in kag_columns:
            # Extraer el nombre base (sin la posición izquierda/derecha)
            if ' I A' in col or ' D A' in col:
                kag_base = col.rsplit(' ', 2)[0]  # Quitar I A o D A
                kag_bases.add(kag_base)
        
        # Para cada nombre base, encontrar sus posiciones izquierda/derecha
        for kag_base in kag_bases:
            kag_left = next((col for col in kag_columns if col.startswith(kag_base) and ' I A' in col), None)
            kag_right = next((col for col in kag_columns if col.startswith(kag_base) and ' D A' in col), None)
            
            if kag_left and kag_right:
                kag_pairs.append((kag_base, kag_left, kag_right))
        
        return kag_pairs
    
    def _extract_station_from_kag(self, kag_name):
        """Extraer información de estación del nombre del aparato de vía"""
        # Intentar extraer la estación del nombre (última parte generalmente)
        parts = kag_name.split()
        if len(parts) >= 3:
            return parts[-1]  # La última parte suele ser la estación
        
        # Si no se puede determinar, devolver desconocido
        return 'Desconocido'
    
    def _process_old_format_file(self, file_path, progress_callback=None):
        """Procesar archivo CSV con formato antiguo"""
        try:
            if progress_callback:
                progress_callback(None, f"Procesando archivo formato antiguo: {os.path.basename(file_path)}")
            
            # Intentar diferentes codificaciones y separadores
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                for separator in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(file_path, sep=separator, encoding=encoding, engine='python')
                        break
                    except:
                        continue
                else:
                    continue
                break
            else:
                if progress_callback:
                    progress_callback(None, f"No se pudo leer el archivo: {os.path.basename(file_path)}")
                return None
            
            # Verificar si contiene columnas de agujas
            ags_columns = [col for col in df.columns if 'AG' in str(col) or 'aguja' in str(col).lower()]
            
            if not ags_columns:
                if progress_callback:
                    progress_callback(None, f"No se encontraron columnas de agujas en: {os.path.basename(file_path)}")
                return None
            
            movements = []
            discordances = []
            
            # Implementar procesamiento específico para formato antiguo
            # Como no tenemos el formato exacto, aquí hay una implementación genérica
            # que deberá adaptarse según el formato real
            
            # Esta es una estructura simplificada para compatibilidad
            return {
                'movements': movements,
                'discordances': discordances
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error procesando archivo formato antiguo: {str(e)}")
            return None
    
    def _process_excel_file(self, file_path, progress_callback=None):
        """Procesar archivo Excel con datos de agujas"""
        try:
            if progress_callback:
                progress_callback(None, f"Procesando archivo Excel: {os.path.basename(file_path)}")
            
            # Intentar leer el archivo Excel
            try:
                df = pd.read_excel(file_path)
            except:
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except:
                    df = pd.read_excel(file_path, engine='xlrd')
            
            # Verificar si contiene columnas de agujas
            ags_columns = [col for col in df.columns if 'AG' in str(col) or 'aguja' in str(col).lower()]
            
            if not ags_columns:
                if progress_callback:
                    progress_callback(None, f"No se encontraron columnas de agujas en: {os.path.basename(file_path)}")
                return None
            
            movements = []
            discordances = []
            
            # Implementar procesamiento específico para formato Excel
            # Como no tenemos el formato exacto, aquí hay una implementación genérica
            # que deberá adaptarse según el formato real
            
            # Esta es una estructura simplificada para compatibilidad
            return {
                'movements': movements,
                'discordances': discordances
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error procesando archivo Excel: {str(e)}")
            return None
    
    def preprocess_data(self, progress_callback=None):
        """Realizar análisis estadístico de tiempos de movimiento"""
        if progress_callback:
            progress_callback(25, "Analizando tiempos de movimiento...")
        
        if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
            # Verificar y asegurar que las columnas necesarias existen
            required_columns = ['Duración (s)', 'Equipo', 'Estación', 'Fecha']
            
            # Si faltan columnas requeridas, intentar adaptarse
            for col in required_columns:
                if col not in self.df_L2_ADV_MOV.columns:
                    if col == 'Duración (s)' and 'Duración' in self.df_L2_ADV_MOV.columns:
                        self.df_L2_ADV_MOV['Duración (s)'] = self.df_L2_ADV_MOV['Duración']
                    else:
                        if progress_callback:
                            progress_callback(None, f"Columna {col} no encontrada, inicializando con valores por defecto")
                        self.df_L2_ADV_MOV[col] = "Desconocido" if col != 'Duración (s)' else 0
            
            # Asegurar que Duración es numérica
            self.df_L2_ADV_MOV['Duración (s)'] = pd.to_numeric(self.df_L2_ADV_MOV['Duración (s)'], errors='coerce')
            
            # Calcular estadísticas de tiempo por equipo
            self.df_L2_ADV_TIME = self.df_L2_ADV_MOV.groupby('Equipo')['Duración (s)'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).reset_index()
            
            # Manejar el caso donde la desviación estándar es NaN
            self.df_L2_ADV_TIME['std'] = self.df_L2_ADV_TIME['std'].fillna(0)
            
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
            
            # Crear una columna ID para compatibilidad con el formato esperado
            hora_inicio_str = self.df_L2_ADV_MOV['Hora Inicio'].astype(str) if 'Hora Inicio' in self.df_L2_ADV_MOV.columns else ''
            self.df_L2_ADV_MOV['ID'] = self.df_L2_ADV_MOV['Fecha'] + '_' + self.df_L2_ADV_MOV['Equipo'] + '_' + hora_inicio_str
            
            # Agregar columna de línea
            self.df_L2_ADV_MOV['Linea'] = 'L2'
        
        if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
            # Verificar y asegurar que las columnas necesarias existen
            required_columns = ['Fecha Hora', 'Equipo', 'Estación']
            
            # Si faltan columnas requeridas, intentar adaptarse
            for col in required_columns:
                if col not in self.df_L2_ADV_DISC.columns:
                    if progress_callback:
                        progress_callback(None, f"Columna {col} no encontrada en discordancias, inicializando con valores por defecto")
                    self.df_L2_ADV_DISC[col] = "Desconocido"
            
            # Crear ID único para cada discordancia
            self.df_L2_ADV_DISC['Fecha Hora'] = pd.to_datetime(self.df_L2_ADV_DISC['Fecha Hora']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Crear una columna ID para compatibilidad con el formato esperado
            self.df_L2_ADV_DISC['ID'] = self.df_L2_ADV_DISC['Fecha Hora'] + '_' + self.df_L2_ADV_DISC['Equipo']
            
            # Crear columna Equipo Estacion para compatibilidad
            self.df_L2_ADV_DISC['Equipo Estacion'] = self.df_L2_ADV_DISC['Equipo'] + '*' + self.df_L2_ADV_DISC['Estación']
            
            # Agregar columna de línea
            self.df_L2_ADV_DISC['Linea'] = 'L2'
        
        if progress_callback:
            progress_callback(40, "Preprocesamiento completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """Detectar aparatos de vía con comportamiento anómalo"""
        if progress_callback:
            progress_callback(70, "Detectando agujas con comportamiento anómalo...")
        
        if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty and 'Anomalía' in self.df_L2_ADV_MOV.columns:
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
                progress_callback(None, f"Se identificaron {len(critical_equipments)} equipos críticos")
        
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
            if 'Hora Inicio' in self.df_L2_ADV_MOV.columns:
                self.df_L2_ADV_MOV['Hora Inicio'] = self.df_L2_ADV_MOV['Hora Inicio'].astype(str)
            
            if 'Hora Fin' in self.df_L2_ADV_MOV.columns:
                self.df_L2_ADV_MOV['Hora Fin'] = self.df_L2_ADV_MOV['Hora Fin'].astype(str)
            
            # Eliminar columnas de análisis estadístico del reporte final
            cols_to_drop = []
            for col in ['mean', 'umbral_superior']:
                if col in self.df_L2_ADV_MOV.columns:
                    cols_to_drop.append(col)
                    
            if cols_to_drop:
                self.df_L2_ADV_MOV = self.df_L2_ADV_MOV.drop(cols_to_drop, axis=1)
            
            # Agregar columna Equipo Estacion para compatibilidad
            if 'Equipo Estacion' not in self.df_L2_ADV_MOV.columns and 'Estación' in self.df_L2_ADV_MOV.columns:
                self.df_L2_ADV_MOV['Equipo Estacion'] = self.df_L2_ADV_MOV['Equipo'] + '*' + self.df_L2_ADV_MOV['Estación']
        
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
                    if 'ID' in df_L2_ADV_DISC_Mensual.columns:
                        df_L2_ADV_DISC_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
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
                    df_L2_ADV_DISC_Mensual = pd.read_csv(mov_file_path)
                    
                    # Concatenar con nuevos datos
                    df_L2_ADV_DISC_Mensual = pd.concat([df_L2_ADV_DISC_Mensual, self.df_L2_ADV_MOV], ignore_index=True)
                    
                    # Eliminar duplicados
                    if 'ID' in df_L2_ADV_DISC_Mensual.columns:
                        df_L2_ADV_DISC_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Guardar reporte actualizado
                    df_L2_ADV_DISC_Mensual.to_csv(mov_file_path, index=False)
                else:
                    # Crear nuevo reporte
                    self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
                
            if progress_callback:
                progress_callback(95, "Reportes actualizados")
            
            return True

        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error actualizando reportes: {str(e)}")
            return False
    def save_dataframe(self):
        """Guardar los dataframe principales"""
        try:
            #guardar dataframe de movimientos
            if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV.csv')
                self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
                
            #guardar dataframe discordancias
            if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC.csv')
                self.df_L2_ADV_DISC.to_csv(disc_file_path, index=False)
                
            #guardar dataframe de estadisticas de tiempo
            
            if self.df_L2_ADV_TIME is not None and not self.df_L2_ADV_TIME.empty:
                time_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_TIME.csv')
                self.df_L2_ADV_TIME.to_csv(time_file_path, index=False)
            
            return True
        except Exception as e:
            print(f"Error al guardar los dataframes: {str(e)}")
            return False
                