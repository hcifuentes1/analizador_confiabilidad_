# processors/cdv_processor_l4.py
import pandas as pd
import numpy as np
import os
import concurrent.futures
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class CDVProcessorL4(BaseProcessor):
    """Procesador para datos CDV de la Línea 4"""
    
    def __init__(self):
        super().__init__(line="L4", analysis_type="CDV")
        # Atributos específicos para CDV L4
        self.df = None
        self.df_L4_2 = None
        self.df_L4_FO = None
        self.df_L4_FL = None
        self.df_L4_OCUP = None
        
        # Factores de umbral para detección de anomalías
        self.f_oc_1 = 0.1
        self.f_lb_2 = 0.05
    
    def find_files(self):
        """Encontrar archivos para análisis CDV de Línea 4"""
        self.csv_files = []
        
        # Verificar si la ruta existe
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Iterar sobre las carpetas dentro de la carpeta raíz
        for folder1 in os.listdir(self.root_folder_path):   
            folder_path1 = os.path.join(self.root_folder_path, folder1)
            if os.path.isdir(folder_path1):
                # Iterar sobre las carpetas dentro de la carpeta actual
                for folder2 in os.listdir(folder_path1):
                    folder_path2 = os.path.join(folder_path1, folder2)
                    if os.path.isdir(folder_path2):
                        # Iterar sobre los archivos dentro de cada carpeta
                        for file in os.listdir(folder_path2):
                            # Verificar si el archivo contiene "VID"
                            if "VID" in file:
                                try:
                                    self.csv_files.append(os.path.join(folder_path2, file))
                                except ValueError:
                                    print(f"Error al analizar la fecha del archivo: {file}")
        
        return len(self.csv_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos para análisis CDV de Línea 4 utilizando procesamiento paralelo"""
        if progress_callback:
            progress_callback(5, f"Leyendo {len(self.csv_files)} archivos...")
        
        df_L4_list = []
        total_files = len(self.csv_files)
        
        # Función para procesar un solo archivo
        def process_file(index_file_tuple):
            index, csv_file = index_file_tuple
            try:
                # Leer el archivo y filtrar por CDV
                df = pd.read_table(csv_file, encoding="Latin-1", sep="|", skiprows=0, header=None, engine='python')
                df = df[df[1].str.contains('TR_CDV', na=False)]
                
                if progress_callback and index % 5 == 0:  # Actualizar cada 5 archivos para no saturar la UI
                    progress = 5 + (index / total_files) * 15
                    progress_callback(progress, f"Procesando archivo {index+1} de {total_files}")
                
                return df
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"No se pudo leer el archivo {csv_file} debido a un error: {e}")
                return pd.DataFrame()  # Retornar DataFrame vacío en caso de error
        
        # Usar ThreadPoolExecutor para procesar archivos en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
            results = list(executor.map(process_file, enumerate(self.csv_files)))
        
        # Filtrar DataFrames vacíos y concatenar resultados
        df_L4_list = [df for df in results if not df.empty]
        
        if df_L4_list:
            if progress_callback:
                progress_callback(20, "Concatenando resultados...")
            
            self.df = pd.concat(df_L4_list, ignore_index=True)
            del df_L4_list  # Liberar memoria
            return True
        else:
            if progress_callback:
                progress_callback(None, "No se encontraron datos válidos en los archivos.")
            return False
    
    def preprocess_data(self, progress_callback=None):
        """Preprocesar datos para análisis CDV de Línea 4"""
        if progress_callback:
            progress_callback(20, "Preprocesando datos...")
        
        # Convertir a datetime y filtrar por fecha (últimos 40 días) - FILTRO TEMPRANO
        self.df[0] = pd.to_datetime(self.df[0], dayfirst=True, errors='coerce')
        date_threshold = datetime.now() - timedelta(days=40)
        self.df = self.df[self.df[0] >= date_threshold]
        
        if progress_callback:
            progress_callback(25, "Procesando columnas...")
        
        # Procesar columnas como en el script original
        df_aux = self.df[1].str.split(":", expand=True)
        df_aux.columns = ["COL.1", "COL.2"]
        self.df = self.df.join(df_aux)
        
        df_aux_2 = self.df["COL.2"].str.split("_", expand=True)
        df_aux_2.columns = ["COL.3", "COL.4"]
        self.df = self.df.join(df_aux_2)
        
        self.df["COL.1"] = self.df["COL.1"].str.replace("__", "_")
        df_aux_3 = self.df["COL.1"].str.split("_", expand=True)
        df_aux_3.columns = ["COL.5", "COL.6", "COL.7", "COL.8", "COL.9"]
        self.df = self.df.join(df_aux_3)
        
        # Seleccionar columnas y renombrar
        cols = [0, 4, 8, 10, 11, 7, 2, 3, 5, 6, 1, 9, 12]
        self.df = self.df.iloc[:, cols]
        self.df = self.df.drop(self.df.columns[[7, 8, 9, 10, 11, 12]], axis=1)
        
        self.df.rename(
            columns={0: "Fecha Hora", "COL.1": "Equipo", "COL.5": "Estacion", 
                    "COL.7": "Subsistema", "COL.8": "Numero Equipo", "COL.4": "Atributo", 2: "Estado"},
            inplace=True
        )
        
        if progress_callback:
            progress_callback(30, "Limpiando datos...")
        
        # Limpieza y preparación adicional
        self.df["Numero Equipo"] = self.df["Numero Equipo"].astype("string")
        self.df = self.df.sort_values(["Equipo", "Fecha Hora"])
        self.df["Fecha Hora"] = pd.to_datetime(self.df["Fecha Hora"], dayfirst=True)
        self.df = self.df.reset_index()
        
        if "index" in self.df.columns:
            self.df = self.df.drop("index", axis=1)
        
        # Eliminar columnas no necesarias
        self.df = self.df.drop(["Subsistema", "Numero Equipo", "Atributo"], axis=1)
        
        # Filtrar por horario operativo (6am a 11pm)
        self.df.set_index('Fecha Hora', inplace=True)
        self.df = self.df[(self.df.index.hour >= 6) & (self.df.index.hour <= 23)]
        self.df = self.df.reset_index()
        
        # Convertir estados "libre" a 1 y "ocupado" a 0
        self.df['Estado'] = self.df['Estado'].replace(' ', '')
        self.df['Estado'] = self.df['Estado'].apply(lambda x: 0 if 'ocupado' in str(x).lower() else x)
        self.df['Estado'] = self.df['Estado'].apply(lambda x: 1 if 'libre' in str(x).lower() else x)
        self.df['Estado'] = self.df['Estado'].replace('ocupado', 0)
        self.df['Estado'] = self.df['Estado'].replace('libre', 1)
        
        # Mantener solo estados válidos y eliminar repeticiones consecutivas
        self.df = self.df[self.df['Estado'].isin([1, 0])]
        self.df["Estado"] = self.df["Estado"].astype("float64")
        self.df["Diff_Aux"] = self.df["Estado"].diff(periods=1)
        self.df["Diff_Aux"] = self.df["Diff_Aux"].astype("string")
        self.df = self.df.loc[~(self.df["Diff_Aux"].str.contains("0.0"))]
        self.df = self.df.drop('Diff_Aux', axis=1)
        
        # Convertir de nuevo a formato texto para facilitar análisis
        self.df['Estado'] = self.df['Estado'].replace(1, 'libre')
        self.df['Estado'] = self.df['Estado'].replace(0, 'ocupado')
        
        # Limpiar nombres de equipos
        self.df["Equipo"] = self.df["Equipo"].str.replace("TR_", "")
        for i in range(10):
            self.df["Equipo"] = self.df["Equipo"].str.replace(f"{i}_", f"{i}")
        
        if progress_callback:
            progress_callback(45, "Preprocesamiento completado")
        
        return True
    
    def calculate_time_differences(self, progress_callback=None):
        """Calcular diferencias de tiempo entre eventos"""
        if progress_callback:
            progress_callback(45, "Calculando diferencias temporales...")
        
        # Diferencias con registros anteriores
        self.df["Diff.Time_-1_row"] = self.df["Fecha Hora"].diff(periods=1)
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].dt.total_seconds()
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].astype("float64")
        self.df["Diff.Time_-1_row"] = round(self.df["Diff.Time_-1_row"], 2)
        
        self.df["Diff.Time_-2_row"] = self.df["Fecha Hora"].diff(periods=2)
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].dt.total_seconds()
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].astype("float64")
        self.df["Diff.Time_-2_row"] = round(self.df["Diff.Time_-2_row"], 2)
        
        # Diferencias con registros siguientes
        self.df["Diff.Time_+1_row"] = self.df["Fecha Hora"].diff(periods=-1)
        self.df["Diff.Time_+1_row"] = -1 * self.df["Diff.Time_+1_row"].dt.total_seconds()
        self.df["Diff.Time_+1_row"] = self.df["Diff.Time_+1_row"].astype("float64")
        self.df["Diff.Time_+1_row"] = round(self.df["Diff.Time_+1_row"], 2)
        
        self.df["Diff.Time_+2_row"] = self.df["Fecha Hora"].diff(periods=-2)
        self.df["Diff.Time_+2_row"] = -1 * self.df["Diff.Time_+2_row"].dt.total_seconds()
        self.df["Diff.Time_+2_row"] = self.df["Diff.Time_+2_row"].astype("float64")
        self.df["Diff.Time_+2_row"] = round(self.df["Diff.Time_+2_row"], 2)
        
        # Filtrar tiempos válidos
        self.df = self.df.loc[self.df["Diff.Time_-1_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_-2_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_+1_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_+2_row"] >= 0.0]
        
        # Calcular tiempo conjunto
        self.df["Tiempo Conjunto"] = self.df["Diff.Time_-1_row"] + self.df["Diff.Time_+2_row"]
        self.df["Tiempo Conjunto"] = self.df["Tiempo Conjunto"].astype("float64")
        self.df["Tiempo Conjunto"] = round(self.df["Tiempo Conjunto"], 2)
        
        if progress_callback:
            progress_callback(60, "Cálculo de diferencias temporales completado")
        
        return True
    
    def calculate_statistics(self, progress_callback=None):
        """Calcular estadísticas para cada tipo de evento"""
        if progress_callback:
            progress_callback(60, "Calculando estadísticas...")
        
        # Crear copias para análisis por separado
        df_L4_lb = self.df.copy()
        df_L4_oc = self.df.copy()
        
        # Estadísticas generales
        df_L4_aux_avg = self.df.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_+1_row",
            aggfunc={np.mean, np.std, np.median, np.min, np.max}
        )
        
        # Estadísticas para estado "ocupado" (liberación)
        df_L4_lb = df_L4_lb.loc[df_L4_lb["Estado"].str.contains("ocupado")]
        df_L4_aux_lb = df_L4_lb.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median, np.min, np.max}
        )
        df_L4_aux_lb.rename(
            columns={"amax": "max_lib", "amin": "min_lib", "mean": "mean_lib", "std": "std_lib", "median": "median_lib"},
            inplace=True
        )
        
        # Estadísticas para estado "libre" (ocupación)
        df_L4_oc = df_L4_oc.loc[df_L4_oc["Estado"].str.contains("libre")]
        df_L4_aux_oc = df_L4_oc.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median, np.min, np.max}
        )
        df_L4_aux_oc.rename(
            columns={"amax": "max_oc", "amin": "min_oc", "mean": "mean_oc", "std": "std_oc", "median": "median_oc"},
            inplace=True
        )
        
        # Combinar estadísticas
        self.df = self.df.merge(df_L4_aux_lb, on="Equipo")
        self.df = self.df.merge(df_L4_aux_oc, on="Equipo")
        
        # Ordenar y eliminar columnas innecesarias
        self.df = self.df.sort_values(["Equipo", "Fecha Hora"])
        
        # Eliminar columnas adicionales que puedan haberse generado
        columns_to_drop = ["min_x", "max_x", "max_y", "min_y"]
        for col in columns_to_drop:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)
        
        # Redondear columnas numéricas
        numeric_columns = [
            "Diff.Time_-1_row", "Diff.Time_-2_row", "Diff.Time_+1_row", "Diff.Time_+2_row",
            "Tiempo Conjunto", "mean_lib", "median_lib", "std_lib", "mean_oc", "median_oc", "std_oc"
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = round(self.df[col], 1)
        
        if progress_callback:
            progress_callback(70, "Cálculo de estadísticas completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """Detectar anomalías basadas en umbrales estadísticos"""
        if progress_callback:
            progress_callback(70, "Detectando anomalías...")
        
        # Detectar Fallos de Ocupación (FO)
        self.df["FO"] = np.where(
            self.df["Estado"].str.contains("ocupado"), 
            np.where(self.df["Diff.Time_+1_row"] < (self.f_oc_1 * self.df["median_oc"]), "PFO", "NFO"), 
            "NA"
        )
        
        # Detectar Fallos de Liberación (FL)
        self.df["FL"] = np.where(
            self.df["Estado"].str.contains("libre"), 
            np.where(self.df["Diff.Time_+1_row"] < (self.f_lb_2 * self.df["median_lib"]), "PFL", "NFL"), 
            "NA"
        )
        
        # Agregar identificador de línea
        self.df["Linea"] = "L4"
        
        if progress_callback:
            progress_callback(80, "Detección de anomalías completada")
        
        return True
    
    def prepare_reports(self, progress_callback=None):
        """Preparar los diferentes reportes"""
        if progress_callback:
            progress_callback(80, "Preparando reportes...")
        
        # 1. Preparar reporte de fallos de ocupación (FO)
        self.df_L4_FO = self.df.loc[self.df['FO'] == 'PFO']
        
        # Eliminar columnas innecesarias
        columns_to_drop = [
            'Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
            'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
            'mean_oc', 'median_oc', 'std_oc', 'FL'
        ]
        self.df_L4_FO = self.df_L4_FO.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L4_FO['Fecha Hora'] = self.df_L4_FO['Fecha Hora'].astype(str)
        self.df_L4_FO['Equipo'] = self.df_L4_FO['Equipo'].astype(str)
        self.df_L4_FO['Diff.Time_+1_row'] = self.df_L4_FO['Diff.Time_+1_row'].astype(str)
        self.df_L4_FO['ID'] = self.df_L4_FO['Fecha Hora'] + self.df_L4_FO['Equipo'] + self.df_L4_FO['Diff.Time_+1_row']
        
        # 2. Preparar reporte de ocupaciones
        self.df['Fecha'] = self.df['Fecha Hora'].dt.date
        
        self.df_L4_OCUP = self.df[self.df['Estado'] == 'ocupado']
        self.df_L4_OCUP = self.df_L4_OCUP.groupby(['Equipo', 'Fecha']).size().reset_index(name='Count')
        
        # Crear ID único
        self.df_L4_OCUP['Fecha'] = self.df_L4_OCUP['Fecha'].astype(str)
        self.df_L4_OCUP['Count'] = self.df_L4_OCUP['Count'].astype(str)
        self.df_L4_OCUP['Equipo'] = self.df_L4_OCUP['Equipo'].astype(str)
        self.df_L4_OCUP['ID'] = self.df_L4_OCUP['Fecha'] + self.df_L4_OCUP['Equipo'] + self.df_L4_OCUP['Count']
        self.df_L4_OCUP['Linea'] = "L4"
        
        # 3. Preparar reporte de fallos de liberación (FL)
        self.df_L4_FL = self.df.loc[self.df['FL'] == 'PFL']
        
        # Eliminar columnas innecesarias
        columns_to_drop = [
            'Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
            'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
            'mean_oc', 'median_oc', 'std_oc', 'FO', 'Fecha'
        ]
        self.df_L4_FL = self.df_L4_FL.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L4_FL['Fecha Hora'] = self.df_L4_FL['Fecha Hora'].astype(str)
        self.df_L4_FL['Equipo'] = self.df_L4_FL['Equipo'].astype(str)
        self.df_L4_FL['Diff.Time_+1_row'] = self.df_L4_FL['Diff.Time_+1_row'].astype(str)
        self.df_L4_FL['ID'] = self.df_L4_FL['Fecha Hora'] + self.df_L4_FL['Equipo'] + self.df_L4_FL['Diff.Time_+1_row']
        
        if progress_callback:
            progress_callback(85, "Preparación de reportes completada")
        
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(85, "Iniciando actualización de reportes...")
            
            # 1. Actualizar reporte de fallos de ocupación
            fo_file_path = os.path.join(self.output_folder_path, 'df_L4_FO_Mensual.csv')
            if os.path.exists(fo_file_path):
                df_L4_FO_Mensual = pd.read_csv(fo_file_path)
                
                # Concatenar y eliminar duplicados
                df_L4_FO_Mensual = pd.concat([df_L4_FO_Mensual, self.df_L4_FO], ignore_index=True)
                df_L4_FO_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L4_FO_Mensual.to_csv(fo_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L4_FO.to_csv(fo_file_path, index=False)
            
            # 2. Actualizar reporte de ocupaciones
            ocup_file_path = os.path.join(self.output_folder_path, 'df_L4_OCUP_Mensual.csv')
            if os.path.exists(ocup_file_path):
                df_L4_OCUP_Mensual = pd.read_csv(ocup_file_path)
                
                # Concatenar y eliminar duplicados
                df_L4_OCUP_Mensual = pd.concat([df_L4_OCUP_Mensual, self.df_L4_OCUP], ignore_index=True)
                df_L4_OCUP_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L4_OCUP_Mensual.to_csv(ocup_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L4_OCUP.to_csv(ocup_file_path, index=False)
            
            # 3. Actualizar reporte de fallos de liberación
            fl_file_path = os.path.join(self.output_folder_path, 'df_L4_FL_Mensual.csv')
            if os.path.exists(fl_file_path):
                df_L4_FL_Mensual = pd.read_csv(fl_file_path)
                
                # Concatenar y eliminar duplicados
                df_L4_FL_Mensual = pd.concat([df_L4_FL_Mensual, self.df_L4_FL], ignore_index=True)
                df_L4_FL_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L4_FL_Mensual.to_csv(fl_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L4_FL.to_csv(fl_file_path, index=False)
            
            if progress_callback:
                progress_callback(95, "Actualización de reportes completada")
            
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al actualizar reportes: {str(e)}")
            return False
    
    def save_dataframe(self):
        """Guardar el DataFrame principal"""
        try:
            # Guardar el DataFrame principal
            main_file_path = os.path.join(self.output_folder_path, 'df_L4_CDV.csv')
            
            # Si el DataFrame es muy grande, guardar en chunks para reducir uso de memoria
            if len(self.df) > 100000:  # Ajustar este número según las necesidades
                chunk_size = 50000     # Tamaño de cada chunk
                for i, chunk in enumerate(np.array_split(self.df, len(self.df) // chunk_size + 1)):
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0    # Solo incluir encabezado en el primer chunk
                    chunk.to_csv(main_file_path, index=True, mode=mode, header=header)
            else:
                # Para DataFrames pequeños o medianos, guardar normalmente
                self.df.to_csv(main_file_path, index=True)
                
            return True
        except Exception as e:
            print(f"Error al guardar DataFrame: {e}")
            return False
    
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos
            if progress_callback:
                progress_callback(0, "Buscando archivos para Línea 4 CDV...")
            num_files = self.find_files()
            if num_files == 0:
                if progress_callback:
                    progress_callback(100, "No se encontraron archivos para procesar")
                return False
            
            # 2. Leer archivos (procesamiento paralelo)
            if progress_callback:
                progress_callback(5, f"Leyendo {num_files} archivos (procesamiento paralelo)...")
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
                progress_callback(60, "Calculando estadísticas...")
            self.calculate_statistics(progress_callback)
            
            # 6. Detección de anomalías
            if progress_callback:
                progress_callback(70, "Detectando anomalías...")
            self.detect_anomalies(progress_callback)
            
            # 7. Preparación de reportes
            if progress_callback:
                progress_callback(80, "Preparando reportes...")
            self.prepare_reports(progress_callback)
            
            # 8. Actualización de reportes existentes
            if progress_callback:
                progress_callback(85, "Actualizando reportes existentes...")
            self.update_reports(progress_callback)
            
            # 9. Guardar DataFrame principal
            if progress_callback:
                progress_callback(95, "Guardando DataFrame principal...")
            self.save_dataframe()
            
            if progress_callback:
                progress_callback(100, "Procesamiento CDV Línea 4 completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False