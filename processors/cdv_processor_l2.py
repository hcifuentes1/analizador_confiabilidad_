# processors/cdv_processor_l2.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class CDVProcessorL2(BaseProcessor):
    """Procesador para datos CDV de la Línea 2"""
    
    def __init__(self):
        super().__init__(line="L2", analysis_type="CDV")
        # Atributos específicos para CDV L2
        self.df_L2_2 = None
        self.df_L2_FO = None
        self.df_L2_FL = None
        self.df_L2_OCUP = None
        
        # Factores de umbral para detección de anomalías
        self.f_oc_1 = 0.1
        self.f_lb_2 = 0.05
        
        # Atributo para el tipo de datos
        self.data_type = "Sacem"  # Valor por defecto
    
    def set_data_type(self, data_type):
        """Establecer tipo de datos (Sacem o SCADA)"""
        self.data_type = data_type
    
    def find_files(self):
        """Encontrar archivos CSV/Excel para análisis CDV de Línea 2"""
        self.data_files = []
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Recorrer carpetas y encontrar archivos CSV o Excel
        for root, dirs, files in os.walk(self.root_folder_path):
            for file in files:
                # Filtrar según el tipo de datos
                if self.data_type == "Sacem":
                    # Buscar archivos de Sacem (actual implementación)
                    if file.endswith(('.xlsx', '.xls', '.csv')):
                        self.data_files.append(os.path.join(root, file))
                elif self.data_type == "SCADA":
                    # En el futuro, aquí se implementará la lógica para archivos SCADA
                    # Por ahora, solo registramos que está en desarrollo
                    continue
        
        return len(self.data_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos CSV/Excel para análisis CDV de Línea 2"""
        df_list = []
        total_files = len(self.data_files)
        
        for i, file_path in enumerate(self.data_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(file_path)}")
                
                # Determinar si es un archivo en el nuevo formato (basado en el nombre o estructura)
                is_new_format = self._is_new_format_file(file_path)
                
                if is_new_format:
                    df = self._process_new_format_file(file_path, progress_callback)
                    if df is not None and not df.empty:
                        df_list.append(df)
                else:
                    # Leer archivo según su extensión (formato anterior)
                    df = self._process_old_format_file(file_path, progress_callback)
                    if df is not None and not df.empty:
                        df_list.append(df)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error en archivo {file_path}: {str(e)}")
        
        if df_list:
            self.df = pd.concat(df_list, ignore_index=True)
            del df_list
            return True
        else:
            return False
    
    def _is_new_format_file(self, file_path):
        """Determinar si un archivo está en el nuevo formato basándose en su estructura"""
        try:
            # Verificar si es un archivo CSV
            if not file_path.lower().endswith('.csv'):
                return False
                
            # Leer las primeras líneas para verificar la estructura
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
            # Verificar si contiene la estructura específica del nuevo formato
            # Por ejemplo, si contiene "ciclo,tiempo,milits" en la primera línea
            if "ciclo,tiempo,milits" in first_line:
                return True
                
            # Intentar leer el archivo como CSV y verificar los nombres de columnas
            try:
                header_df = pd.read_csv(file_path, nrows=1)
                column_names = header_df.columns.tolist()
                
                # Verificar si tiene las columnas específicas del nuevo formato
                if (any("SigV" in col for col in column_names) and 
                    any("SigA" in col for col in column_names) and
                    any("CDV" in col for col in column_names)):
                    return True
            except:
                pass
                
            return False
            
        except Exception:
            # Si hay algún error al intentar leer el archivo, asumimos que no es el nuevo formato
            return False
    
    def _process_new_format_file(self, file_path, progress_callback=None):
        """Procesar archivo en el nuevo formato"""
        try:
            if progress_callback:
                progress_callback(None, f"Detectado archivo en nuevo formato: {os.path.basename(file_path)}")
            
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
            # Reemplazamos error_bad_lines por on_bad_lines para compatibilidad con pandas recientes
            try:
                # Para versiones recientes de pandas (1.3.0+)
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8', on_bad_lines='warn')
            except TypeError:
                # Para versiones anteriores de pandas
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
            
            # Manejar caso donde el archivo podría tener una sola columna con todos los datos
            if len(df.columns) == 1:
                # Intentar dividir la única columna usando diferentes separadores
                for possible_sep in [',', ';', '\t']:
                    try:
                        # Intentamos ambas formas para manejar diferentes versiones de pandas
                        try:
                            df = pd.read_csv(file_path, sep=possible_sep, encoding='utf-8', on_bad_lines='warn')
                        except TypeError:
                            df = pd.read_csv(file_path, sep=possible_sep, encoding='utf-8', error_bad_lines=False)
                        
                        if len(df.columns) > 1:
                            break
                    except:
                        continue
            
            # Verificar si el DataFrame contiene las columnas esperadas
            if 'ciclo' in df.columns and 'tiempo' in df.columns:
                # Este es el nuevo formato con columnas "ciclo" y "tiempo"
                
                # Crear columna de fecha y hora combinada
                if 'tiempo' in df.columns:
                    # Especificar formato para evitar warning
                    try:
                        df['Fecha Hora'] = pd.to_datetime(df['tiempo'], format='%d/%m/%Y %H:%M:%S')
                    except:
                        # Si falla con el formato específico, intentar inferir el formato
                        df['Fecha Hora'] = pd.to_datetime(df['tiempo'], errors='coerce')
                else:
                    # Si no existe 'tiempo', intentamos con otras columnas
                    if 'ciclo' in df.columns and any('milits' in col for col in df.columns):
                        time_col = next(col for col in df.columns if 'milits' in col)
                        # Especificar formato para evitar warning si es posible
                        df['Fecha Hora'] = pd.to_datetime(df['ciclo'] + " " + df[time_col], errors='coerce')
                
                # Procesar columnas de CDV
                cdv_columns = [col for col in df.columns if 'CDV' in col and 'SigA' not in col]
                if not cdv_columns:
                    # Si no hay columnas explícitamente marcadas como CDV, buscar columnas con patrones de valores binarios
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    cdv_columns = [col for col in numeric_cols if set(df[col].dropna().unique()).issubset({0, 1})]
                
                # Preparar DataFrame para formato "derretido" (melted)
                data_rows = []
                
                for _, row in df.iterrows():
                    fecha_hora = row['Fecha Hora']
                    
                    # Si la fecha es inválida, continuamos con la siguiente fila
                    if pd.isna(fecha_hora):
                        continue
                        
                    for cdv_col in cdv_columns:
                        # Solo procesamos si el valor no es nulo
                        if not pd.isna(row[cdv_col]):
                            # Determinar estado
                            estado_valor = row[cdv_col]
                            estado = 'Liberacion' if estado_valor == 1 else 'Ocupacion' if estado_valor == 0 else 'Desconocido'
                            
                            # Extraer información de estación del nombre de la columna
                            estacion = self._extract_station_from_column_name(cdv_col)
                            
                            # Agregar fila al conjunto de datos
                            data_rows.append({
                                'Fecha Hora': fecha_hora,
                                'Equipo': cdv_col,
                                'Estacion': estacion,
                                'Subsistema': 'CDV',
                                'Estado': estado
                            })
                
                if data_rows:
                    return pd.DataFrame(data_rows)
                else:
                    return None
            else:
                # No es el formato esperado
                return None
                
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al procesar archivo en nuevo formato {file_path}: {str(e)}")
            return None
    
    def _extract_station_from_column_name(self, column_name):
        """Extraer información de estación del nombre de columna"""
        # Este método intenta extraer la estación del nombre de una columna
        # Por ejemplo, de "CDV 12 AV" extraería "12"
        
        # Patrón para extraer números de la columna
        import re
        station_match = re.search(r'(?:CDV|SigA)\s*(\d+)', column_name)
        if station_match:
            return station_match.group(1)
        
        # Patrón alternativo
        station_match = re.search(r'(\d+)\s*(?:AV|CV)', column_name)
        if station_match:
            return station_match.group(1)
            
        # Si no se puede extraer, devolvemos un valor genérico
        return 'NA'
    
    def _process_old_format_file(self, file_path, progress_callback=None):
        """Procesar archivo en el formato anterior"""
        try:
            # Leer archivo según su extensión
            if file_path.lower().endswith('.csv'):
                # Probar diferentes codificaciones y separadores comunes para CSV
                try:
                    # Primero intentar con separador punto y coma (común en configuraciones europeas)
                    try:
                        # Para versiones recientes de pandas (1.3.0+)
                        df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='warn')
                    except TypeError:
                        # Para versiones anteriores de pandas
                        df = pd.read_csv(file_path, sep=';', encoding='utf-8', error_bad_lines=False)
                except:
                    try:
                        # Luego intentar con separador coma (estándar)
                        try:
                            df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='warn')
                        except TypeError:
                            df = pd.read_csv(file_path, sep=',', encoding='utf-8', error_bad_lines=False)
                    except:
                        try:
                            # Intentar con codificación Latin-1
                            try:
                                df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='warn')
                            except TypeError:
                                df = pd.read_csv(file_path, sep=';', encoding='latin1', error_bad_lines=False)
                        except:
                            # Último intento con coma y Latin-1
                            try:
                                df = pd.read_csv(file_path, sep=',', encoding='latin1', on_bad_lines='warn')
                            except TypeError:
                                df = pd.read_csv(file_path, sep=',', encoding='latin1', error_bad_lines=False)
            else:
                # Para archivos Excel
                try:
                    # Primero intentar con pandas directamente
                    df = pd.read_excel(file_path)
                except Exception as e1:
                    # Si falla, intentar con engine='openpyxl'
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    except Exception as e2:
                        # Último intento con engine='xlrd'
                        df = pd.read_excel(file_path, engine='xlrd')
            
            # Verificar si es un archivo con formato esperado
            if 'FECHA' in df.columns and 'HORA' in df.columns:
                # Filtrar columnas que empiezan con 'CDV'
                # Filtrar columnas que empiezan con 'CDV' y no contienen 'SigA'
                cdv_cols = [col for col in df.columns if col.startswith('CDV') and 'SigA' not in col]
                if not cdv_cols:
                    return None
                
                # Tomar sólo las columnas necesarias
                cols_to_keep = ['ciclo', 'FECHA', 'HORA'] + cdv_cols
                cols_to_keep = [col for col in cols_to_keep if col in df.columns]
                df = df[cols_to_keep]
                
                # Crear una versión "derretida" (melted) del dataframe para análisis
                id_vars = ['ciclo', 'FECHA', 'HORA']
                id_vars = [col for col in id_vars if col in df.columns]
                
                df_melted = pd.melt(
                    df, 
                    id_vars=id_vars, 
                    value_vars=cdv_cols,
                    var_name='Equipo',
                    value_name='Estado'
                )
                
                # Agregar columna de estación basada en el nombre del CDV
                df_melted['Estacion'] = df_melted['Equipo'].str.extract(r'CDV\s+(\d+)')
                df_melted['Subsistema'] = 'CDV'
                
                # Crear columna de fecha y hora combinada
                # Especificar formato para evitar warning
                try:
                    # Primero intentamos con un formato específico
                    format_str = None
                    
                    # Verificar el formato de la fecha
                    fecha_sample = str(df_melted['FECHA'].iloc[0]) if not df_melted.empty else ""
                    hora_sample = str(df_melted['HORA'].iloc[0]) if not df_melted.empty else ""
                    
                    # Determinar formato basado en muestras
                    if '/' in fecha_sample:
                        format_str = '%d/%m/%Y %H:%M:%S'
                    elif '-' in fecha_sample:
                        format_str = '%Y-%m-%d %H:%M:%S'
                    
                    if format_str:
                        df_melted['Fecha Hora'] = pd.to_datetime(
                            df_melted['FECHA'].astype(str) + ' ' + df_melted['HORA'].astype(str),
                            format=format_str, 
                            errors='coerce'
                        )
                    else:
                        df_melted['Fecha Hora'] = pd.to_datetime(
                            df_melted['FECHA'].astype(str) + ' ' + df_melted['HORA'].astype(str),
                            errors='coerce'
                        )
                except:
                    # Si falla, usamos el método predeterminado
                    df_melted['Fecha Hora'] = pd.to_datetime(
                        df_melted['FECHA'].astype(str) + ' ' + df_melted['HORA'].astype(str),
                        errors='coerce'
                    )
                
                # Modificar Estado: considerar 1 como "Liberacion" y 0 como "Ocupacion"
                # Se utiliza infer_objects para evitar el warning futuro
                # Opción 1: Para versiones más recientes de pandas
                if hasattr(df_melted, 'infer_objects'):
                    df_melted['Estado'] = df_melted['Estado'].map(
                        lambda x: 'Ocupacion' if x == 0 else 'Liberacion' if x == 1 else 'Desconocido'
                    ).infer_objects(copy=False)
                else:
                    # Opción 2: Para versiones antiguas de pandas
                    df_melted['Estado'] = df_melted['Estado'].map(
                        lambda x: 'Ocupacion' if x == 0 else 'Liberacion' if x == 1 else 'Desconocido'
                    )
                
                # Seleccionar columnas finales
                df_melted = df_melted[['Fecha Hora', 'Equipo', 'Estacion', 'Subsistema', 'Estado']]
                
                return df_melted
            
            return None
                
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al procesar archivo en formato original {file_path}: {str(e)}")
            return None
    
    def preprocess_data(self, progress_callback=None):
        """Realizar el preprocesamiento inicial de los datos"""
        if progress_callback:
            progress_callback(20, "Iniciando preprocesamiento de datos...")
        
        # Ordenar por equipo y fecha/hora
        self.df = self.df.sort_values(["Equipo", "Fecha Hora"])
        
        if progress_callback:
            progress_callback(25, "Filtrando por fecha...")
            
        # Filtrar por fecha (últimos 40 días)
        date_threshold = datetime.now() - timedelta(days=40)
        self.df = self.df[self.df["Fecha Hora"] >= date_threshold]
        
        if progress_callback:
            progress_callback(30, "Filtrando por hora del día...")
            
        # Filtrar por hora del día (operación del metro, de 6am a 11pm)
        self.df.set_index('Fecha Hora', inplace=True)
        self.df = self.df[(self.df.index.hour >= 6) & (self.df.index.hour <= 23)]
        self.df = self.df.reset_index()
        
        if progress_callback:
            progress_callback(35, "Procesando estados...")
            
        # Procesar estados
        self.process_states()
        
        if progress_callback:
            progress_callback(45, "Preprocesamiento completado")
        
        return True
    
    def process_states(self):
        """Procesar estados para CDV de Línea 2"""
        # Crear copia del DataFrame
        self.df_L2_2 = self.df.copy()
        
        # Convertir estados a numéricos usando infer_objects para evitar warnings
        if hasattr(self.df_L2_2, 'infer_objects'):
            # Para versiones más recientes de pandas
            self.df_L2_2['Estado'] = self.df_L2_2['Estado'].replace('Liberacion', 1).infer_objects(copy=False)
            self.df_L2_2['Estado'] = self.df_L2_2['Estado'].replace('Ocupacion', 0).infer_objects(copy=False)
        else:
            # Para versiones antiguas de pandas
            self.df_L2_2['Estado'] = self.df_L2_2['Estado'].replace('Liberacion', 1)
            self.df_L2_2['Estado'] = self.df_L2_2['Estado'].replace('Ocupacion', 0)
        
        # Ordenar y filtrar
        self.df_L2_2 = self.df_L2_2.sort_values(["Equipo", "Fecha Hora"])
        self.df_L2_2 = self.df_L2_2[self.df_L2_2['Estado'].isin([1, 0])]
        self.df_L2_2["Estado"] = self.df_L2_2["Estado"].astype("float64")
        
        # Detectar cambios de estado
        self.df_L2_2["Diff_Aux"] = self.df_L2_2.groupby("Equipo")["Estado"].diff(periods=1)
        self.df_L2_2["Diff_Aux"] = self.df_L2_2["Diff_Aux"].astype("string")
        self.df_L2_2 = self.df_L2_2.loc[~(self.df_L2_2["Diff_Aux"].str.contains("0.0"))]
        
        # Limpiar y restaurar etiquetas originales
        self.df_L2_2 = self.df_L2_2.drop('Diff_Aux', axis=1)
        self.df_L2_2['Estado'] = self.df_L2_2['Estado'].replace(1, 'Liberacion')
        self.df_L2_2['Estado'] = self.df_L2_2['Estado'].replace(0, 'Ocupacion')
        
        # Actualizar DataFrame principal
        self.df = self.df_L2_2.copy()
    
    def calculate_time_differences(self, progress_callback=None):
        """Calcular diferencias de tiempo entre registros"""
        if progress_callback:
            progress_callback(45, "Calculando diferencias temporales...")
            
        # Calcular diferencia con registros anteriores (por equipo)
        self.df["Diff.Time_-1_row"] = self.df.groupby("Equipo")["Fecha Hora"].diff(periods=1)
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].dt.total_seconds()
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].astype("float64")
        self.df["Diff.Time_-1_row"] = round(self.df["Diff.Time_-1_row"], 1)
        
        self.df["Diff.Time_-2_row"] = self.df.groupby("Equipo")["Fecha Hora"].diff(periods=2)
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].dt.total_seconds()
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].astype("float64")
        self.df["Diff.Time_-2_row"] = round(self.df["Diff.Time_-2_row"], 1)
        
        if progress_callback:
            progress_callback(50, "Calculando diferencias temporales hacia adelante...")
            
        # Calcular diferencia con registros siguientes (por equipo)
        self.df["Diff.Time_+1_row"] = -1 * self.df.groupby("Equipo")["Fecha Hora"].diff(periods=-1)
        self.df["Diff.Time_+1_row"] = self.df["Diff.Time_+1_row"].dt.total_seconds()
        self.df["Diff.Time_+1_row"] = self.df["Diff.Time_+1_row"].astype("float64")
        self.df["Diff.Time_+1_row"] = round(self.df["Diff.Time_+1_row"], 1)
        
        self.df["Diff.Time_+2_row"] = -1 * self.df.groupby("Equipo")["Fecha Hora"].diff(periods=-2)
        self.df["Diff.Time_+2_row"] = self.df["Diff.Time_+2_row"].dt.total_seconds()
        self.df["Diff.Time_+2_row"] = self.df["Diff.Time_+2_row"].astype("float64")
        self.df["Diff.Time_+2_row"] = round(self.df["Diff.Time_+2_row"], 1)
        
        if progress_callback:
            progress_callback(55, "Filtrando datos por tiempos válidos...")
            
        # Filtrar por tiempos positivos
        self.df = self.df.loc[self.df["Diff.Time_-1_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_-2_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_+1_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_+2_row"] >= 0.0]
        
        if progress_callback:
            progress_callback(60, "Calculando tiempo conjunto...")
            
        # Calcular tiempo conjunto
        self.df["Tiempo Conjunto"] = self.df["Diff.Time_-1_row"] + (self.df["Diff.Time_+2_row"])
        self.df["Tiempo Conjunto"] = self.df["Tiempo Conjunto"].astype("float64")
        self.df["Tiempo Conjunto"] = round(self.df["Tiempo Conjunto"], 2)
        
        if progress_callback:
            progress_callback(65, "Cálculo de diferencias temporales completado")
        
        return True
    
    def calculate_statistics(self, progress_callback=None):
        """Calcular estadísticas para los estados de ocupación y liberación"""
        if progress_callback:
            progress_callback(65, "Iniciando cálculo de estadísticas...")
            
        # Crear copias para análisis separados
        df_L2_lb = self.df.copy()
        df_L2_oc = self.df.copy()
        
        if progress_callback:
            progress_callback(67, "Calculando estadísticas generales...")
            
        # Calcular estadísticas generales
        df_L2_aux_avg = self.df.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_+1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        
        if progress_callback:
            progress_callback(69, "Calculando estadísticas para liberación...")
            
        # Estadísticas para liberación
        df_L2_lb = df_L2_lb.loc[df_L2_lb["Estado"].str.contains("Ocupacion")]
        df_L2_aux_lb = df_L2_lb.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        df_L2_aux_lb.rename(
            columns={"mean": "mean_lib", "std": "std_lib", "median": "median_lib"},
            inplace=True
        )
        
        if progress_callback:
            progress_callback(71, "Calculando estadísticas para ocupación...")
            
        # Estadísticas para ocupación
        df_L2_oc = df_L2_oc.loc[df_L2_oc["Estado"].str.contains("Liberacion")]
        df_L2_aux_oc = df_L2_oc.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        df_L2_aux_oc.rename(
            columns={"mean": "mean_oc", "std": "std_oc", "median": "median_oc"},
            inplace=True
        )
        
        if progress_callback:
            progress_callback(73, "Combinando estadísticas...")
            
        # Combinar estadísticas con el DataFrame principal
        self.df = self.df.merge(df_L2_aux_lb, on="Equipo")
        self.df = self.df.merge(df_L2_aux_oc, on="Equipo")
        
        # Ordenar los datos
        self.df = self.df.sort_values(["Equipo", "Fecha Hora"])
        
        # Redondear valores estadísticos
        for col in ["Diff.Time_-1_row", "Diff.Time_-2_row", "Diff.Time_+1_row", 
                   "Diff.Time_+2_row", "Tiempo Conjunto", "mean_lib", "median_lib", 
                   "std_lib", "mean_oc", "median_oc", "std_oc"]:
            self.df[col] = round(self.df[col], 1)
        
        if progress_callback:
            progress_callback(75, "Cálculo de estadísticas completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """Detectar anomalías basadas en umbrales estadísticos"""
        if progress_callback:
            progress_callback(75, "Iniciando detección de anomalías...")
            
        # Detectar Fallos de Ocupación (FO)
        self.df["FO"] = np.where(
            (self.df["Estado"].str.contains("Ocupacion")), 
            np.where((self.df["Diff.Time_+1_row"] < (self.f_oc_1 * self.df["median_oc"])), "PFO", "NFO"), 
            "NA"
        )
        
        if progress_callback:
            progress_callback(80, "Detectando fallos de liberación...")
            
        # Detectar Fallos de Liberación (FL)
        self.df["FL"] = np.where(
            (self.df["Estado"].str.contains("Liberacion")), 
            np.where((self.df["Diff.Time_+1_row"] < (self.f_lb_2 * self.df["median_lib"])), "PFL", "NFL"), 
            "NA"
        )
        
        # Agregar identificador de línea
        self.df["Linea"] = "L2"
        
        if progress_callback:
            progress_callback(85, "Detección de anomalías completada")
        
        return True
    
    def prepare_reports(self, progress_callback=None):
        """Preparar los diferentes reportes"""
        if progress_callback:
            progress_callback(85, "Iniciando preparación de reportes...")
            
        # 1. Preparar reporte de fallos de ocupación (FO)
        self.df_L2_FO = self.df.loc[self.df['FO'] == 'PFO']
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
                           'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
                           'mean_oc', 'median_oc', 'std_oc', 'FL']
        self.df_L2_FO = self.df_L2_FO.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L2_FO['Fecha Hora'] = self.df_L2_FO['Fecha Hora'].astype(str)
        self.df_L2_FO['Equipo'] = self.df_L2_FO['Equipo'].astype(str)
        self.df_L2_FO['Diff.Time_+1_row'] = self.df_L2_FO['Diff.Time_+1_row'].astype(str)
        self.df_L2_FO['ID'] = self.df_L2_FO['Fecha Hora'] + self.df_L2_FO['Equipo'] + self.df_L2_FO['Diff.Time_+1_row']
        
        if progress_callback:
            progress_callback(87, "Preparando reporte de ocupaciones...")
            
        # 2. Preparar reporte de conteo de ocupaciones
        # Extraer fecha (sin hora)
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha Hora']).dt.date
        
        # Filtrar y agrupar
        self.df_L2_OCUP = self.df[self.df['Estado'] == 'Ocupacion']
        self.df_L2_OCUP = self.df_L2_OCUP.groupby(['Equipo', 'Fecha']).size().reset_index(name='Count')
        
        # Crear ID único
        self.df_L2_OCUP['Fecha'] = self.df_L2_OCUP['Fecha'].astype(str)
        self.df_L2_OCUP['Count'] = self.df_L2_OCUP['Count'].astype(str)
        self.df_L2_OCUP['Equipo'] = self.df_L2_OCUP['Equipo'].astype(str)
        self.df_L2_OCUP['ID'] = self.df_L2_OCUP['Fecha'] + self.df_L2_OCUP['Equipo'] + self.df_L2_OCUP['Count']
        
        if progress_callback:
            progress_callback(89, "Preparando reporte de fallos de liberación...")
            
        # 3. Preparar reporte de fallos de liberación (FL)
        self.df_L2_FL = self.df.loc[self.df['FL'] == 'PFL']
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
                          'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
                          'mean_oc', 'median_oc', 'std_oc', 'FO', 'Fecha']
        self.df_L2_FL = self.df_L2_FL.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L2_FL['Fecha Hora'] = self.df_L2_FL['Fecha Hora'].astype(str)
        self.df_L2_FL['Equipo'] = self.df_L2_FL['Equipo'].astype(str)
        self.df_L2_FL['Diff.Time_+1_row'] = self.df_L2_FL['Diff.Time_+1_row'].astype(str)
        self.df_L2_FL['ID'] = self.df_L2_FL['Fecha Hora'] + self.df_L2_FL['Equipo'] + self.df_L2_FL['Diff.Time_+1_row']
        
        if progress_callback:
            progress_callback(90, "Preparación de reportes completada")
        
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(90, "Iniciando actualización de reportes...")
            
            # 1. Actualizar reporte de fallos de ocupación
            fo_file_path = os.path.join(self.output_folder_path, 'df_L2_FO_Mensual.csv')
            if os.path.exists(fo_file_path):
                df_L2_FO_Mensual = pd.read_csv(fo_file_path)
                
                # Concatenar y eliminar duplicados
                df_L2_FO_Mensual = pd.concat([df_L2_FO_Mensual, self.df_L2_FO], ignore_index=True)
                df_L2_FO_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L2_FO_Mensual.to_csv(fo_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L2_FO.to_csv(fo_file_path, index=False)
            
            if progress_callback:
                progress_callback(93, "Actualizando reporte de ocupaciones...")
                
            # 2. Actualizar reporte de conteo de ocupaciones
            ocup_file_path = os.path.join(self.output_folder_path, 'df_L2_OCUP_Mensual.csv')
            if os.path.exists(ocup_file_path):
                df_L2_OCUP_Mensual = pd.read_csv(ocup_file_path)
                
                # Concatenar y eliminar duplicados
                df_L2_OCUP_Mensual = pd.concat([df_L2_OCUP_Mensual, self.df_L2_OCUP], ignore_index=True)
                df_L2_OCUP_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L2_OCUP_Mensual.to_csv(ocup_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L2_OCUP.to_csv(ocup_file_path, index=False)
            
            if progress_callback:
                progress_callback(96, "Actualizando reporte de fallos de liberación...")
                
            # 3. Actualizar reporte de fallos de liberación
            fl_file_path = os.path.join(self.output_folder_path, 'df_L2_FL_Mensual.csv')
            if os.path.exists(fl_file_path):
                df_L2_FL_Mensual = pd.read_csv(fl_file_path)
                
                # Concatenar y eliminar duplicados
                df_L2_FL_Mensual = pd.concat([df_L2_FL_Mensual, self.df_L2_FL], ignore_index=True)
                df_L2_FL_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L2_FL_Mensual.to_csv(fl_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L2_FL.to_csv(fl_file_path, index=False)
            
            if progress_callback:
                progress_callback(98, "Actualización de reportes completada")
            
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al actualizar reportes: {str(e)}")
            return False
    
    def save_dataframe(self):
        """Guardar el DataFrame principal"""
        try:
            # Guardar el DataFrame principal
            main_file_path = os.path.join(self.output_folder_path, 'df_L2_CDV.csv')
            self.df.to_csv(main_file_path, index=True)
            return True
        except Exception as e:
            return False
    
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos
            if progress_callback:
                progress_callback(0, "Buscando archivos para Línea 2 CDV...")
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
            
            # 3. Preprocesamiento
            if progress_callback:
                progress_callback(20, "Preprocesando datos...")
            self.preprocess_data(progress_callback)
            
            # 4. Cálculo de diferencias temporales
            if progress_callback:
                progress_callback(45, "Calculando diferencias temporales...")
            self.calculate_time_differences(progress_callback)
            
            # 5. Cálculo de estadísticas
            if progress_callback:
                progress_callback(65, "Calculando estadísticas...")
            self.calculate_statistics(progress_callback)
            
            # 6. Detección de anomalías
            if progress_callback:
                progress_callback(75, "Detectando anomalías...")
            self.detect_anomalies(progress_callback)
            
            # 7. Preparación de reportes
            if progress_callback:
                progress_callback(85, "Preparando reportes...")
            self.prepare_reports(progress_callback)
            
            # 8. Actualización de reportes existentes
            if progress_callback:
                progress_callback(90, "Actualizando reportes existentes...")
            self.update_reports(progress_callback)
            
            # 9. Guardar DataFrame principal
            if progress_callback:
                progress_callback(98, "Guardando DataFrame principal...")
            self.save_dataframe()
            
            if progress_callback:
                progress_callback(100, "Procesamiento CDV Línea 2 completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False