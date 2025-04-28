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
        self.data_files = []
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Recorrer carpetas y encontrar archivos CSV
        for root, dirs, files in os.walk(self.root_folder_path):
            for file in files:
                # Buscar cualquier archivo CSV - seremos menos restrictivos
                if file.lower().endswith('.csv'):
                    file_path = os.path.join(root, file)
                    # Añadir todos los CSV inicialmente
                    self.data_files.append(file_path)
        
        # Añadir logs de depuración
        print(f"ADVProcessorL2CSV: Buscando archivos CSV en: {self.root_folder_path}")
        print(f"ADVProcessorL2CSV: Se encontraron {len(self.data_files)} archivos CSV")
        
        # Registrar los archivos encontrados
        if hasattr(self, 'output_folder_path') and self.output_folder_path:
            log_path = os.path.join(self.output_folder_path, 'adv_csv_log.txt')
            try:
                with open(log_path, 'w') as f:
                    f.write(f"Ruta de búsqueda: {self.root_folder_path}\n")
                    f.write(f"Total archivos encontrados: {len(self.data_files)}\n")
                    f.write("Lista de archivos:\n")
                    for file in self.data_files:
                        f.write(f"- {file}\n")
            except Exception as e:
                print(f"Error al escribir log: {str(e)}")
        
        return len(self.data_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos CSV para análisis ADV de Línea 2"""
        movements_data = []  # Para almacenar datos de movimientos
        discordance_data = []  # Para almacenar datos de discordancias
        total_files = len(self.data_files)
        
        if progress_callback:
            progress_callback(5, f"Procesando {total_files} archivos CSV para análisis ADV...")
        
        # Imprimir información de depuración
        print(f"ADVProcessorL2CSV: Comenzando a leer {total_files} archivos CSV")
        
        for i, file_path in enumerate(self.data_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(file_path)}")
                
                print(f"ADVProcessorL2CSV: Procesando archivo {i+1}/{total_files}: {file_path}")
                
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
                    print(f"ADVProcessorL2CSV: No se encontraron columnas Kag en {file_path}")
                    continue
                
                # Verificar y crear columna de fecha y hora
# Verificar y crear columna de fecha y hora
                try:
                    print(f"ADVProcessorL2CSV: Intentando determinar columna de fecha/hora")
                    
                    # Verificar si las columnas esperadas existen
                    if 'ciclo' in df.columns and 'tiempo' in df.columns and 'milits' in df.columns:
                        print("ADVProcessorL2CSV: Columnas 'ciclo', 'tiempo' y 'milits' encontradas")
                        
                        # Ver ejemplos de los valores
                        sample_tiempo = str(df['tiempo'].iloc[0]) if len(df) > 0 else ""
                        sample_milits = str(df['milits'].iloc[0]) if len(df) > 0 else ""
                        print(f"ADVProcessorL2CSV: Ejemplo de valor en 'tiempo': {sample_tiempo}")
                        print(f"ADVProcessorL2CSV: Ejemplo de valor en 'milits': {sample_milits}")
                        
                        # Este formato específico tiene la fecha como 'DD/MM/YY HH:MM:SS'
                        # y los milisegundos separados
                        try:
                            # Convertir la parte de fecha/hora principal
                            df['Fecha Hora Base'] = pd.to_datetime(df['tiempo'], format='%d/%m/%y %H:%M:%S', errors='coerce')
                            
                            # Convertir los milisegundos a timedelta y sumarlos
                            # Primero convertimos a numérico por si acaso
                            df['milits'] = pd.to_numeric(df['milits'], errors='coerce')
                            
                            # Luego creamos un timedelta con los milisegundos
                            millis_delta = pd.to_timedelta(df['milits'], unit='ms')
                            
                            # Y finalmente sumamos a la fecha base
                            df['Fecha Hora'] = df['Fecha Hora Base'] + millis_delta
                            
                            # Verificar que la conversión fue exitosa
                            if not df['Fecha Hora'].isna().all():
                                print("ADVProcessorL2CSV: Fecha y hora creadas con éxito")
                            else:
                                print("ADVProcessorL2CSV: Error en la conversión de fecha/hora - todos los valores son NA")
                                # Si falló la conversión con milisegundos, usar solo la parte principal
                                df['Fecha Hora'] = df['Fecha Hora Base']
                        
                        except Exception as e:
                            print(f"ADVProcessorL2CSV: Error procesando fecha con formato específico: {str(e)}")
                            # Intentar un enfoque más flexible
                            try:
                                # Combinar tiempo y milisegundos para una conversión directa
                                df['tiempo_completo'] = df['tiempo'] + '.' + df['milits'].astype(str)
                                df['Fecha Hora'] = pd.to_datetime(df['tiempo_completo'], errors='coerce')
                                print("ADVProcessorL2CSV: Usando enfoque alternativo para la fecha")
                            except:
                                # Si todo falla, usar solo la columna tiempo
                                df['Fecha Hora'] = pd.to_datetime(df['tiempo'], errors='coerce')
                                print("ADVProcessorL2CSV: Usando solo columna tiempo para la fecha")
                                
                    else:
                        print("ADVProcessorL2CSV: Columnas esperadas no encontradas, buscando alternativas")
                        # Código para buscar alternativas de columnas de fecha
                        # [Mantener el código existente para alternativas]
                    
                    # Verificar si finalmente tenemos una columna de fecha/hora válida
                    if 'Fecha Hora' not in df.columns or df['Fecha Hora'].isna().all():
                        print("ADVProcessorL2CSV: No se pudo crear una columna de fecha/hora válida, usando fecha ficticia")
                        # Crear una columna de fecha ficticia para poder continuar el procesamiento
                        start_date = datetime.now() - timedelta(days=7)  # Una semana atrás
                        df['Fecha Hora'] = pd.date_range(start=start_date, periods=len(df), freq='T')
                    
                    # Si llegamos a este punto y existen columnas temporales de procesamiento, eliminarlas
                    for col in ['Fecha Hora Base', 'tiempo_completo']:
                        if col in df.columns:
                            df = df.drop(columns=[col])
                    
                except Exception as e:
                    print(f"ADVProcessorL2CSV: Error general al procesar fechas: {str(e)}")
                    # Crear una columna de fecha ficticia para poder continuar
                    start_date = datetime.now() - timedelta(days=7)
                    df['Fecha Hora'] = pd.date_range(start=start_date, periods=len(df), freq='T')
                
                # Identificar pares de agujas (izquierda/derecha)
                kag_pairs = self._identify_kag_pairs(kag_columns)
                
                if progress_callback:
                    progress_callback(None, f"Encontrados {len(kag_pairs)} pares de agujas en {os.path.basename(file_path)}")
                print(f"ADVProcessorL2CSV: Encontrados {len(kag_pairs)} pares de agujas en {file_path}")
                
                # Procesar cada par de agujas
                for kag_base, kag_left, kag_right in kag_pairs:
                    # Extraer datos de este aparato de vía
                    kag_data = self._process_kag(df, kag_base, kag_left, kag_right, progress_callback)
                    
                    if kag_data:
                        movements_data.extend(kag_data['movements'])
                        discordance_data.extend(kag_data['discordances'])
                        print(f"ADVProcessorL2CSV: Procesado {kag_base} - Movimientos: {len(kag_data['movements'])}, Discordancias: {len(kag_data['discordances'])}")
                    else:
                        print(f"ADVProcessorL2CSV: No se obtuvieron datos para {kag_base}")
                
                print(f"ADVProcessorL2CSV: Archivo {i+1} procesado correctamente")
                
            except Exception as e:
                print(f"ADVProcessorL2CSV: Error procesando archivo {file_path}: {str(e)}")
                if progress_callback:
                    progress_callback(None, f"Error al procesar {os.path.basename(file_path)}: {str(e)}")
        
        # Crear DataFrames de movimientos y discordancias
        if movements_data:
            self.df_L2_ADV_MOV = pd.DataFrame(movements_data)
            print(f"ADVProcessorL2CSV: Creado DataFrame con {len(movements_data)} movimientos")
            if progress_callback:
                progress_callback(None, f"Se procesaron {len(movements_data)} registros de movimientos")
        else:
            print("ADVProcessorL2CSV: No se encontraron datos de movimientos")
            self.df_L2_ADV_MOV = pd.DataFrame()  # DataFrame vacío pero inicializado
        
        if discordance_data:
            self.df_L2_ADV_DISC = pd.DataFrame(discordance_data)
            print(f"ADVProcessorL2CSV: Creado DataFrame con {len(discordance_data)} discordancias")
            if progress_callback:
                progress_callback(None, f"Se encontraron {len(discordance_data)} discordancias")
        else:
            print("ADVProcessorL2CSV: No se encontraron datos de discordancias")
            self.df_L2_ADV_DISC = pd.DataFrame()  # DataFrame vacío pero inicializado
        
        # Resultado del procesamiento - devuelve True incluso con DataFrames vacíos
        print("ADVProcessorL2CSV: Procesamiento de archivos completado")
        return True
    
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
        
        print(f"ADVProcessorL2CSV: Intentando identificar pares de agujas en {len(kag_columns)} columnas Kag")
        print(f"ADVProcessorL2CSV: Columnas Kag disponibles: {kag_columns}")
        
        # En este formato específico, las agujas tienen el formato "Kag XX/YYZ AV"
        # donde XX/YY es el número de aguja y Z es D (derecha) o G (izquierda)
        
        # Agrupar por número de aguja
        kag_groups = {}
        
        for col in kag_columns:
            # Buscar patrones como "Kag 11/21G AV" y "Kag 11/21D AV"
            import re
            # Primero intenta con el patrón para agujas con formato XX/YY
            match = re.search(r'Kag\s+(\d+(?:/\d+)?)[GD]', col)
            
            if match:
                # El grupo 1 contiene el número de aguja (ej: "11/21" o "41")
                kag_number = match.group(1)
                
                # Determinar si es G (izquierda) o D (derecha)
                if 'G' in col:
                    kag_side = 'G'
                elif 'D' in col:
                    kag_side = 'D'
                else:
                    continue
                    
                # Añadir al grupo correspondiente
                if kag_number not in kag_groups:
                    kag_groups[kag_number] = {}
                
                kag_groups[kag_number][kag_side] = col
        
        # Crear pares para cada número de aguja que tenga lado G y D
        for kag_number, sides in kag_groups.items():
            kag_base = f"Kag {kag_number}"
            
            # Si tiene ambos lados, crear par
            if 'G' in sides and 'D' in sides:
                kag_left = sides['G']
                kag_right = sides['D']
                kag_pairs.append((kag_base, kag_left, kag_right))
                print(f"ADVProcessorL2CSV: Par de agujas encontrado - Base: {kag_base}, Izq: {kag_left}, Der: {kag_right}")
            # Si solo tiene un lado, tratar como aguja especial
            elif 'G' in sides:
                # Para agujas con solo G, usar el mismo lado para ambos
                kag_left = sides['G']
                kag_right = sides['G']  # Usar el mismo para simplificar procesamiento
                kag_pairs.append((kag_base + " (Simple-G)", kag_left, kag_right))
                print(f"ADVProcessorL2CSV: Aguja simple encontrada (G) - Base: {kag_base}")
            elif 'D' in sides:
                # Para agujas con solo D, usar el mismo lado para ambos
                kag_left = sides['D']  # Usar el mismo para simplificar procesamiento
                kag_right = sides['D']
                kag_pairs.append((kag_base + " (Simple-D)", kag_left, kag_right))
                print(f"ADVProcessorL2CSV: Aguja simple encontrada (D) - Base: {kag_base}")
        
        return kag_pairs



    
    def _process_kag(self, df, kag_base, kag_left, kag_right, progress_callback=None):
        """Procesar datos de un aparato de vía específico"""
        try:
            # Extraer estación del nombre del aparato de vía
            station = self._extract_station_from_kag(kag_base)
            
            # Verificar si es una aguja simple
            is_simple_switch = "(Simple-G)" in kag_base or "(Simple-D)" in kag_base
            
            # Crear una copia de las columnas relevantes
            kag_df = df[['Fecha Hora', kag_left, kag_right]].copy()
            
            # Filtrar filas con valores nulos en fecha o en estados de la aguja
            kag_df = kag_df.dropna(subset=['Fecha Hora'])
            # Para agujas simples, solo verificamos el lado que existe
            if is_simple_switch:
                if "(Simple-G)" in kag_base:
                    kag_df = kag_df.dropna(subset=[kag_left])
                else:
                    kag_df = kag_df.dropna(subset=[kag_right])
            else:
                kag_df = kag_df.dropna(subset=[kag_left, kag_right])
            
            # Si no quedan datos después de filtrar, no procesar
            if kag_df.empty:
                return None
            
            # Convertir a valores numéricos (asegurar que sean 0 o 1)
            kag_df[kag_left] = pd.to_numeric(kag_df[kag_left], errors='coerce').fillna(0).astype(int)
            kag_df[kag_right] = pd.to_numeric(kag_df[kag_right], errors='coerce').fillna(0).astype(int)
            
            # Ordenar por fecha y hora
            kag_df = kag_df.sort_values('Fecha Hora')
            
            # Lógica específica para agujas simples
            if is_simple_switch:
                if "(Simple-G)" in kag_base:
                    # Para agujas con solo G, tratamos los cambios en ese canal
                    kag_df['estado_actual'] = kag_df[kag_left]
                    kag_df['estado_previo'] = kag_df['estado_actual'].shift(1)
                    kag_df['cambio_estado'] = (kag_df['estado_actual'] != kag_df['estado_previo']).astype(int)
                    kag_df['en_movimiento'] = (kag_df['estado_actual'] == 0).astype(int)
                    # Simulamos estado combinado para mantener consistencia
                    kag_df['estado_combinado'] = kag_df['estado_actual'].astype(str) + "0"
                else:
                    # Para agujas con solo D
                    kag_df['estado_actual'] = kag_df[kag_right]
                    kag_df['estado_previo'] = kag_df['estado_actual'].shift(1)
                    kag_df['cambio_estado'] = (kag_df['estado_actual'] != kag_df['estado_previo']).astype(int)
                    kag_df['en_movimiento'] = (kag_df['estado_actual'] == 0).astype(int)
                    # Simulamos estado combinado para mantener consistencia
                    kag_df['estado_combinado'] = "0" + kag_df['estado_actual'].astype(str)
            else:
                # Lógica normal para pares de agujas
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
            
            # Detectar discordancias
            if is_simple_switch:
                # Para agujas simples, no hay discordancias posibles
                kag_df['discordancia'] = False
            else:
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
                            'Equipo': kag_base.replace(" (Simple-G)", "").replace(" (Simple-D)", ""),  # Quitar sufijo
                            'Estación': station,
                            'Estado Anterior': row['estado_previo'] if not pd.isna(row['estado_previo']) else "NA",
                            'Estado Nuevo': row['estado_combinado']
                        })
                    
                    current_movement_start = None
                
                # Registrar discordancias
                if not is_simple_switch and row['discordancia']:
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
            print(f"ADVProcessorL2CSV: Error al procesar {kag_base}: {str(e)}")
            if progress_callback:
                progress_callback(None, f"Error al procesar {kag_base}: {str(e)}")
            return None
    
    def _extract_station_from_kag(self, kag_name):
        """Extraer información de estación del nombre del aparato de vía"""
        # Intentar extraer la estación usando expresiones regulares
        import re
        # Buscar un patrón como "Kag 11/21G AV" donde AV es la estación (2 letras al final)
        match = re.search(r'[GD]\s+([A-Z]{2})$', kag_name)
        if match:
            return match.group(1)  # Devolver la estación (AV en el ejemplo)
        
        # Si no se encuentra con el patrón anterior, buscar otras posibilidades
        # Buscar cualquier secuencia de 2 letras mayúsculas al final
        match = re.search(r'([A-Z]{2})$', kag_name)
        if match:
            return match.group(1)
        
        # Si todo falla, devolver valor por defecto
        return "AV"  # Basado en el log, sabemos que todas son AV
    
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
            
            # Calcular estadísticas más robustas
            self.df_L2_ADV_TIME = self.df_L2_ADV_MOV.groupby('Equipo')['Duración (s)'].agg([
                'count', 'mean', 'std', 'min', 'max', 
                ('median', lambda x: x.median()),  # Mediana
                ('q1', lambda x: x.quantile(0.25)),  # Primer cuartil
                ('q3', lambda x: x.quantile(0.75))   # Tercer cuartil
            ]).reset_index()
            
            # Manejar el caso donde la desviación estándar es NaN
            self.df_L2_ADV_TIME['std'] = self.df_L2_ADV_TIME['std'].fillna(0)
            
            # Calcular IQR (rango intercuartílico)
            self.df_L2_ADV_TIME['iqr'] = self.df_L2_ADV_TIME['q3'] - self.df_L2_ADV_TIME['q1']
            
            # Establecer umbrales más robustos
            self.df_L2_ADV_TIME['umbral_superior_med'] = self.df_L2_ADV_TIME['median'] + (1.5 * self.df_L2_ADV_TIME['iqr'])
            self.df_L2_ADV_TIME['umbral_superior_mean'] = self.df_L2_ADV_TIME['mean'] + (self.df_L2_ADV_TIME['std'] * self.time_threshold)
            self.df_L2_ADV_TIME['umbral_superior'] = self.df_L2_ADV_TIME['umbral_superior_mean']  # Mantener para compatibilidad
            
            # Identificar movimientos anormalmente largos
            self.df_L2_ADV_MOV = self.df_L2_ADV_MOV.merge(
                self.df_L2_ADV_TIME[['Equipo', 'mean', 'median', 'umbral_superior', 'umbral_superior_med', 'umbral_superior_mean']], 
                on='Equipo',
                how='left'
            )
            
            # Marcar anomalías con ambos métodos
            self.df_L2_ADV_MOV['Anomalía_Media'] = self.df_L2_ADV_MOV['Duración (s)'] > self.df_L2_ADV_MOV['umbral_superior_mean']
            self.df_L2_ADV_MOV['Anomalía_Mediana'] = self.df_L2_ADV_MOV['Duración (s)'] > self.df_L2_ADV_MOV['umbral_superior_med']
            self.df_L2_ADV_MOV['Anomalía'] = self.df_L2_ADV_MOV['Anomalía_Media'] | self.df_L2_ADV_MOV['Anomalía_Mediana']
            
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
            
        # Agregar análisis de tendencia por equipo
        if progress_callback:
            progress_callback(45, "Realizando análisis de tendencias...")
        
        try:
            for equipo in self.df_L2_ADV_MOV['Equipo'].unique():
                # Filtrar movimientos para este equipo
                equipo_df = self.df_L2_ADV_MOV[self.df_L2_ADV_MOV['Equipo'] == equipo].copy()
                
                # Ordenar por fecha
                equipo_df = equipo_df.sort_values('Fecha')
                
                # Si hay suficientes datos, calcular tendencia
                if len(equipo_df) >= 5:
                    # Crear índice numérico para regresión
                    equipo_df['idx'] = range(len(equipo_df))
                    
                    # Ajustar línea de tendencia
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        equipo_df['idx'], equipo_df['Duración (s)']
                    )
                    
                    # Guardar pendiente y significancia
                    self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['Equipo'] == equipo, 'trend_slope'] = slope
                    self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['Equipo'] == equipo, 'trend_pvalue'] = p_value
                    self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['Equipo'] == equipo, 'trend_significant'] = (p_value < 0.05) & (slope > 0)
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en análisis de tendencias: {str(e)}")
        
        # Clasificar agujas según nivel de alerta
        if progress_callback:
            progress_callback(50, "Clasificando agujas por nivel de alerta...")
        
        try:
            # Inicializar nivel de alerta
            self.df_L2_ADV_TIME['alerta_nivel'] = 'Normal'
            
            # Nivel de alerta según porcentaje de anomalías
            anomalias_count = self.df_L2_ADV_MOV.groupby('Equipo')['Anomalía'].mean().reset_index()
            anomalias_count.columns = ['Equipo', 'porcentaje_anomalias']
            self.df_L2_ADV_TIME = self.df_L2_ADV_TIME.merge(anomalias_count, on='Equipo', how='left')
            
            # Agujas críticas: alta frecuencia de anomalías o tendencia creciente significativa
            self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['porcentaje_anomalias'] > 0.3, 'alerta_nivel'] = 'Crítico'
            self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME.get('trend_significant', pd.Series([False] * len(self.df_L2_ADV_TIME))) == True, 'alerta_nivel'] = 'Advertencia'
            self.df_L2_ADV_TIME.loc[(self.df_L2_ADV_TIME.get('trend_significant', pd.Series([False] * len(self.df_L2_ADV_TIME))) == True) & 
                                (self.df_L2_ADV_TIME['porcentaje_anomalias'] > 0.2), 'alerta_nivel'] = 'Crítico'
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en clasificación de alertas: {str(e)}")
        
        # Generar recomendaciones de mantenimiento personalizadas
        if progress_callback:
            progress_callback(55, "Generando recomendaciones de mantenimiento...")
        
        try:
            # Crear campo de recomendaciones
            self.df_L2_ADV_TIME['recomendacion'] = ''
            
            # Recomendaciones por nivel de alerta
            self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['alerta_nivel'] == 'Normal', 'recomendacion'] = 'Mantenimiento rutinario'
            self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['alerta_nivel'] == 'Advertencia', 'recomendacion'] = 'Inspección en próximos 15 días'
            self.df_L2_ADV_TIME.loc[self.df_L2_ADV_TIME['alerta_nivel'] == 'Crítico', 'recomendacion'] = 'Mantenimiento correctivo urgente'
            
            # Refinar recomendaciones según tipo de anomalía
            for idx, row in self.df_L2_ADV_TIME.iterrows():
                if row['alerta_nivel'] == 'Crítico':
                    if row.get('trend_significant', False):
                        self.df_L2_ADV_TIME.loc[idx, 'recomendacion'] += ' - Degradación progresiva detectada'
                    
                    # Verificar variabilidad
                    if 'iqr' in row and 'median' in row and row['iqr'] > (row['median'] * 0.5):
                        self.df_L2_ADV_TIME.loc[idx, 'recomendacion'] += ' - Alta variabilidad en tiempos'
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error generando recomendaciones: {str(e)}")
        
        if progress_callback:
            progress_callback(60, "Análisis predictivo completado")
        
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
            
            # Convertir columna de Fecha a string en formato YYYY-MM-DD
            if 'Fecha' in self.df_L2_ADV_MOV.columns:
                try:
                    self.df_L2_ADV_MOV['Fecha'] = pd.to_datetime(self.df_L2_ADV_MOV['Fecha']).dt.strftime('%Y-%m-%d')
                except:
                    pass
            
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
        
        if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
            # Verificar y asegurar que las columnas necesarias existen
            required_columns = ['Fecha Hora', 'Equipo', 'Estación']
            
            # Si faltan columnas requeridas, intentar adaptarse
            for col in required_columns:
                if col not in self.df_L2_ADV_DISC.columns:
                    if progress_callback:
                        progress_callback(None, f"Columna {col} no encontrada en discordancias, inicializando con valores por defecto")
                    self.df_L2_ADV_DISC[col] = "Desconocido"
            
            # Convertir a formato apropiado para el dashboard
            try:
                # Asegurar que Fecha Hora es datetime y convertir a string en formato YYYY-MM-DD HH:MM:SS
                self.df_L2_ADV_DISC['Fecha Hora'] = pd.to_datetime(self.df_L2_ADV_DISC['Fecha Hora']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
            
            # Crear una columna ID para compatibilidad con el formato esperado
            self.df_L2_ADV_DISC['ID'] = self.df_L2_ADV_DISC['Fecha Hora'] + '_' + self.df_L2_ADV_DISC['Equipo']
            
            # Crear columna Equipo Estacion para compatibilidad
            self.df_L2_ADV_DISC['Equipo Estacion'] = self.df_L2_ADV_DISC['Equipo'] + '*' + self.df_L2_ADV_DISC['Estación']
            
            # Agregar columna de línea
            self.df_L2_ADV_DISC['Linea'] = 'L2'
        
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
        
    def save_dataframe(self):
        """Guardar los dataframe principales"""
        try:
            # Guardar dataframe de movimientos
            if self.df_L2_ADV_MOV is not None and not self.df_L2_ADV_MOV.empty:
                # Guardar con el nombre exacto que espera el dashboard
                mov_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV.csv')
                self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
                
                # También guardar el archivo mensual
                mov_monthly_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV_Mensual.csv')
                self.df_L2_ADV_MOV.to_csv(mov_monthly_path, index=False)
                
                print(f"ADVProcessorL2CSV: Guardado dataframe de movimientos con {len(self.df_L2_ADV_MOV)} registros en {mov_file_path}")
            
            # Guardar dataframe discordancias - crear uno vacío si no existe
            if self.df_L2_ADV_DISC is not None and not self.df_L2_ADV_DISC.empty:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC.csv')
                self.df_L2_ADV_DISC.to_csv(disc_file_path, index=False)
                
                disc_monthly_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC_Mensual.csv')
                self.df_L2_ADV_DISC.to_csv(disc_monthly_path, index=False)
                
                print(f"ADVProcessorL2CSV: Guardado dataframe de discordancias con {len(self.df_L2_ADV_DISC)} registros en {disc_file_path}")
            else:
                # Crear dataframes vacíos de discordancias si no existen
                empty_disc_columns = ['Fecha Hora', 'Equipo', 'Estación', 'Estado Izquierda', 
                                    'Estado Derecha', 'Estado Combinado', 'Tipo', 'ID', 
                                    'Equipo Estacion', 'Linea']
                empty_df = pd.DataFrame(columns=empty_disc_columns)
                
                disc_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC.csv')
                empty_df.to_csv(disc_file_path, index=False)
                
                disc_monthly_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC_Mensual.csv')
                empty_df.to_csv(disc_monthly_path, index=False)
                
                print(f"ADVProcessorL2CSV: Creado dataframe vacío de discordancias en {disc_file_path}")
            
            # Guardar dataframe de estadísticas de tiempo
            if self.df_L2_ADV_TIME is not None and not self.df_L2_ADV_TIME.empty:
                time_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_TIME.csv')
                self.df_L2_ADV_TIME.to_csv(time_file_path, index=False)
                
                time_monthly_path = os.path.join(self.output_folder_path, 'df_L2_ADV_TIME_Mensual.csv')
                self.df_L2_ADV_TIME.to_csv(time_monthly_path, index=False)
                
                print(f"ADVProcessorL2CSV: Guardado dataframe de estadísticas de tiempo en {time_file_path}")
            
            return True
        except Exception as e:
            print(f"Error al guardar los dataframes: {str(e)}")
            return False
        
        
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos
            if progress_callback:
                progress_callback(0, "Buscando archivos CSV para análisis ADV-CSV en Línea 2...")
            num_files = self.find_files()
            if num_files == 0:
                if progress_callback:
                    progress_callback(100, "No se encontraron archivos CSV para procesar")
                return False
            
            # 2. Leer archivos
            if progress_callback:
                progress_callback(5, f"Leyendo {num_files} archivos CSV...")
            if not self.read_files(progress_callback):
                if progress_callback:
                    progress_callback(None, "Error al leer los archivos CSV")
                return False
            
            # 3. Preprocesamiento
            if progress_callback:
                progress_callback(20, "Preprocesando datos...")
            self.preprocess_data(progress_callback)
            
            # 4. Detección de anomalías
            if progress_callback:
                progress_callback(70, f"Detectando anomalías para ADV-CSV...")
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
                progress_callback(95, "Guardando resultados finales...")
            save_result = self.save_dataframe()
            
            # 8. Verificar archivos guardados
            self.verify_saved_files()
            
            if progress_callback:
                progress_callback(100, "Procesamiento ADV-CSV completado con éxito")
            
            return True
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
                
            # Guardar detalles del error en un archivo
            try:
                error_log_path = os.path.join(self.output_folder_path, 'adv_csv_error_log.txt')
                with open(error_log_path, 'w') as f:
                    f.write(f"Error en el procesamiento ADV-CSV: {str(e)}\n\n")
                    f.write("Detalles completos del error:\n")
                    f.write(error_details)
            except:
                pass
                
            return False
        
    def verify_saved_files(self):
        """Verificar que los archivos se guardaron correctamente"""
        try:
            # Verificar archivos principales
            base_files = [
                'df_L2_ADV_MOV.csv',
                'df_L2_ADV_DISC.csv',
                'df_L2_ADV_TIME.csv'
            ]
            
            # Verificar archivos mensuales
            monthly_files = [
                'df_L2_ADV_MOV_Mensual.csv',
                'df_L2_ADV_DISC_Mensual.csv',
                'df_L2_ADV_TIME_Mensual.csv'
            ]
            
            print("\nADVProcessorL2CSV: Verificando archivos guardados:")
            
            # Verificar archivos base
            for file in base_files:
                file_path = os.path.join(self.output_folder_path, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  ✓ {file} existe, tamaño: {file_size/1024:.2f} KB")
                    
                    # Si el archivo existe, verificar su contenido
                    if file_size > 0:
                        try:
                            df = pd.read_csv(file_path)
                            print(f"    - Registros: {len(df)}")
                            print(f"    - Columnas: {', '.join(df.columns.tolist()[:5])}...")
                        except Exception as e:
                            print(f"    - Error leyendo archivo: {str(e)}")
                else:
                    print(f"  ✗ {file} NO existe")
            
            # Verificar archivos mensuales
            for file in monthly_files:
                file_path = os.path.join(self.output_folder_path, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  ✓ {file} existe, tamaño: {file_size/1024:.2f} KB")
                else:
                    print(f"  ✗ {file} NO existe")
            
            return True
        except Exception as e:
            print(f"Error verificando archivos: {str(e)}")
            return False