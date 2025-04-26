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
                # Buscar archivos CSV que puedan contener datos de agujas (Kag)
                if file.lower().endswith('.csv'):
                    # Verificar si el archivo contiene datos de Kag leyendo las primeras líneas
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            header = f.readline()
                            # Buscar indicadores de datos de agujas en la cabecera
                            if any(indicator in header for indicator in ['Kag', 'kag', 'AGS', 'AG_', 'aguja']):
                                self.csv_files.append(os.path.join(root, file))
                                continue
                            
                            # Si no encontramos en la cabecera, verificar en las primeras 5 líneas
                            for _ in range(5):
                                line = f.readline()
                                if any(indicator in line for indicator in ['Kag', 'kag', 'AGS', 'AG_', 'aguja']):
                                    self.csv_files.append(os.path.join(root, file))
                                    break
                    except:
                        # Si hay error al leer, intentar añadir el archivo de todos modos
                        # ya que podría ser un problema de codificación pero el archivo podría ser válido
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
                separator = self._detect_separator(file_path)
                
                # Leer el archivo CSV con el separador detectado
                try:
                    # Para versiones recientes de pandas (1.3.0+)
                    df = pd.read_csv(file_path, sep=separator, encoding='utf-8', on_bad_lines='warn')
                except TypeError:
                    # Para versiones anteriores de pandas
                    df = pd.read_csv(file_path, sep=separator, encoding='utf-8', error_bad_lines=False)
                
                # Si la lectura falla, intentar con otras codificaciones
                if df.empty or len(df.columns) <= 1:
                    for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
                            if not df.empty and len(df.columns) > 1:
                                break
                        except:
                            continue
                
                # Verificar si el archivo tiene columnas relacionadas con agujas (Kag)
                kag_columns = [col for col in df.columns if 'Kag' in col]
                
                if not kag_columns:
                    if progress_callback:
                        progress_callback(None, f"No se encontraron columnas Kag en {os.path.basename(file_path)}")
                    continue
                
                # Verificar y crear columna de fecha y hora si existe 'tiempo' o 'ciclo'
                if 'tiempo' in df.columns:
                    df['Fecha Hora'] = pd.to_datetime(df['tiempo'], errors='coerce')
                elif 'ciclo' in df.columns and 'milits' in df.columns:
                    df['Fecha Hora'] = pd.to_datetime(df['ciclo'] + ' ' + df['milits'], errors='coerce')
                elif 'FECHA' in df.columns and 'HORA' in df.columns:
                    df['Fecha Hora'] = pd.to_datetime(df['FECHA'] + ' ' + df['HORA'], errors='coerce')
                else:
                    # Si no hay columnas de tiempo reconocibles, intentar usar la primera columna
                    first_col = df.columns[0]
                    if df[first_col].dtype == 'object' and ':' in str(df[first_col].iloc[0]):
                        df['Fecha Hora'] = pd.to_datetime(df[first_col], errors='coerce')
                
                if 'Fecha Hora' not in df.columns or df['Fecha Hora'].isna().all():
                    if progress_callback:
                        progress_callback(None, f"No se pudo determinar la columna de fecha y hora en {os.path.basename(file_path)}")
                    continue
                
                # Identificar pares de agujas (izquierda/derecha)
                kag_pairs = self._identify_kag_pairs(kag_columns)
                
                if progress_callback:
                    progress_callback(None, f"Encontrados {len(kag_pairs)} pares de agujas en {os.path.basename(file_path)}")
                
                # Procesar cada par de agujas
                for kag_base, kag_left, kag_right in kag_pairs:
                    # Extraer datos de este aparato de vía
                    kag_data = self._process_kag(df, kag_base, kag_left, kag_right, progress_callback)
                    
                    if kag_data:
                        movements_data.extend(kag_data['movements'])
                        discordance_data.extend(kag_data['discordances'])
            
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error al procesar {os.path.basename(file_path)}: {str(e)}")
        
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
        return self.df_L2_ADV_MOV is not None or self.df_L2_ADV_DISC is not None
    
    def _detect_separator(self, file_path):
        """Detectar el separador usado en el archivo CSV"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
            # Contar ocurrencias de posibles separadores
            separators = {',': first_line.count(','), 
                          ';': first_line.count(';'), 
                          '\t': first_line.count('\t')}
            
            # Encontrar el separador más frecuente
            best_separator = max(separators, key=separators.get)
            
            return best_separator if separators[best_separator] > 0 else ','
        except:
            # Valor por defecto si hay error
            return ','
    
    def _identify_kag_pairs(self, kag_columns):
        """Identificar pares de aparatos de vía (izquierda/derecha)"""
        kag_pairs = []
        
        # Patrones para identificar posición de agujas
        left_patterns = ['I', 'I A', 'izq', 'izquierda', 'left']
        right_patterns = ['D', 'D A', 'der', 'derecha', 'right']
        
        # Extraer los nombres base de los aparatos de vía
        kag_bases = set()
        for col in kag_columns:
            # Buscar patrones para extraer el nombre base
            for left_pattern in left_patterns:
                if left_pattern in col:
                    # Quitar el patrón de izquierda para obtener el nombre base
                    kag_base = col.replace(left_pattern, '').strip()
                    if any(kag_base in k for k in kag_columns):
                        kag_bases.add(kag_base)
                        break
            
            # Si no se encontró con patrones de izquierda, intentar con números de Kag
            if 'Kag' in col:
                try:
                    # Extraer el número de Kag (patrón típico: Kag xx/xxG)
                    kag_num = col.split('Kag')[1].strip().split()[0]
                    kag_base = f"Kag {kag_num}"
                    kag_bases.add(kag_base)
                except:
                    pass
        
        # Para cada nombre base, encontrar sus posiciones izquierda/derecha
        for kag_base in kag_bases:
            # Buscar columnas para este Kag
            matching_cols = [col for col in kag_columns if kag_base in col]
            
            kag_left = None
            kag_right = None
            
            # Buscar por patrones específicos
            for col in matching_cols:
                if any(pattern in col for pattern in left_patterns):
                    kag_left = col
                elif any(pattern in col for pattern in right_patterns):
                    kag_right = col
            
            # Si no se encontraron por patrones, intentar por posición en el nombre
            if not kag_left or not kag_right:
                for col in matching_cols:
                    if 'G' in col or 'I' in col:
                        kag_left = col
                    elif 'D' in col:
                        kag_right = col
            
            # Si se encontraron ambos, añadir el par
            if kag_left and kag_right:
                kag_pairs.append((kag_base, kag_left, kag_right))
        
        return kag_pairs
    
    def _process_kag(self, df, kag_base, kag_left, kag_right, progress_callback=None):
        """Procesar datos de un aparato de vía específico"""
        try:
            # Extraer estación del nombre del aparato de vía
            station = self._extract_station_from_kag(kag_base)
            
            # Crear una copia de las columnas relevantes
            kag_df = df[['Fecha Hora', kag_left, kag_right]].copy()
            
            # Filtrar filas con valores nulos en fecha o en estados de la aguja
            kag_df = kag_df.dropna(subset=['Fecha Hora', kag_left, kag_right])
            
            # Si no quedan datos después de filtrar, no procesar
            if kag_df.empty:
                return None
            
            # Convertir a valores numéricos (asegurar que sean 0 o 1)
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
            kag_df['inicio_movimiento'] = (kag_df['en_movimiento'] == 1) & ((kag_df['en_movimiento'].shift(1) == 0) | (pd.isna(kag_df['en_movimiento'].shift(1))))
            kag_df['fin_movimiento'] = (kag_df['en_movimiento'] == 0) & (kag_df['en_movimiento'].shift(1) == 1)
            
            # Detectar discordancias (ambos en 1 o transiciones inválidas)
            kag_df['discordancia'] = (
                # Ambos en 1 (inválido para agujas)
                (kag_df[kag_left] == 1) & (kag_df[kag_right] == 1) |
                # Transiciones directas sin pasar por movimiento
                ((kag_df['estado_combinado'] == '10') & (kag_df['estado_previo'] == '01') & ~kag_df['inicio_movimiento']) |
                ((kag_df['estado_combinado'] == '01') & (kag_df['estado_previo'] == '10') & ~kag_df['inicio_movimiento'])
            )
            
            # Recolectar datos de movimientos
            movements = []
            current_movement_start = None
            
            # Datos de discordancias
            discordances = []
            
            for idx, row in kag_df.iterrows():
                # Registrar inicio de movimiento
                if row['inicio_movimiento']:
                    current_movement_start = row['Fecha Hora']
                
                # Registrar fin de movimiento y calcular duración
                if row['fin_movimiento'] and current_movement_start is not None:
                    movement_duration = (row['Fecha Hora'] - current_movement_start).total_seconds()
                    
                    # Solo registrar movimientos razonables (mayores a 0.1 segundos y menos de 3 minutos)
                    if 0.1 <= movement_duration <= 180:
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
                    tipo_discordancia = 'Ambos lados activos' if row[kag_left] == 1 and row[kag_right] == 1 else 'Transición directa'
                    discordances.append({
                        'Fecha Hora': row['Fecha Hora'],
                        'Equipo': kag_base,
                        'Estación': station,
                        'Estado Izquierda': row[kag_left],
                        'Estado Derecha': row[kag_right],
                        'Estado Combinado': row['estado_combinado'],
                        'Tipo': tipo_discordancia
                    })
            
            return {
                'movements': movements,
                'discordances': discordances
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al procesar {kag_base}: {str(e)}")
            return None
    
    def _extract_station_from_kag(self, kag_name):
        """Extraer información de estación del nombre del aparato de vía"""
        # Buscar estación al final del nombre (patrón común)
        parts = kag_name.split()
        if len(parts) >= 2:
            # La estación suele ser de 2 letras al final
            last_part = parts[-1]
            if len(last_part) <= 3 and last_part.isalpha():
                return last_part
            
            # También podría estar en otros formatos como "Kag 11/21G A"
            # donde A es la estación
            for part in reversed(parts):
                if len(part) <= 3 and part.isalpha():
                    return part
        
        # Intentar extraer estación de patrones conocidos
        import re
        station_match = re.search(r'[A-Z]{2,3}$', kag_name)
        if station_match:
            return station_match.group(0)
        
        # Si no se encuentra ningún patrón claro, usar un valor genérico
        return 'ST'
    
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
                    df_L2_ADV_MOV_Mensual = pd.read_csv(mov_file_path)
                    
                    # Concatenar con nuevos datos
                    df_L2_ADV_MOV_Mensual = pd.concat([df_L2_ADV_MOV_Mensual, self.df_L2_ADV_MOV], ignore_index=True)
                    
                    # Eliminar duplicados
                    if 'ID' in df_L2_ADV_MOV_Mensual.columns:
                        df_L2_ADV_MOV_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Guardar reporte actualizado
                    df_L2_ADV_MOV_Mensual.to_csv(mov_file_path, index=False)
                else:
                    # Crear nuevo reporte
                    self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
                
            # 3. Actualizar reporte de estadísticas de tiempo
            if self.df_L2_ADV_TIME is not None and not self.df_L2_ADV_TIME.empty:
                time_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_TIME_Mensual.csv')
                
                if os.path.exists(time_file_path):
                    # Cargar reporte existente
                    df_L2_ADV_TIME_Mensual = pd.read_csv(time_file_path)
                    
                    # Concatenar con nuevos datos
                    df_L2_ADV_TIME_Mensual = pd.concat([df_L2_ADV_TIME_Mensual, self.df_L2_ADV_TIME], ignore_index=True)
                    
                    # Agrupar por equipo y calcular estadísticas combinadas
                    df_L2_ADV_TIME_Mensual = df_L2_ADV_TIME_Mensual.groupby('Equipo').agg({
                        'count': 'sum',
                        'mean': 'mean',
                        'std': lambda x: np.sqrt(np.mean(x**2)),  # Estimación de desviación estándar combinada
                        'min': 'min',
                        'max': 'max'
                    }).reset_index()
                    
                    # Recalcular umbrales
                    df_L2_ADV_TIME_Mensual['umbral_superior'] = df_L2_ADV_TIME_Mensual['mean'] + (df_L2_ADV_TIME_Mensual['std'] * self.time_threshold)
                    
                    # Guardar reporte actualizado
                    df_L2_ADV_TIME_Mensual.to_csv(time_file_path, index=False)
                else:
                    # Crear nuevo reporte
                    self.df_L2_ADV_TIME.to_csv(time_file_path, index=False)
            
            if progress_callback:
                progress_callback(95, "Reportes actualizados")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al actualizar reportes: {str(e)}")
            return False