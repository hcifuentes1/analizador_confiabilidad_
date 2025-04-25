# processors/adv_processor_l4.py
import pandas as pd
import numpy as np
import os
import concurrent.futures
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class ADVProcessorL4(BaseProcessor):
    """Procesador para datos ADV (Agujas) de la Línea 4"""
    
    def __init__(self):
        super().__init__(line="L4", analysis_type="ADV")
        # Atributos específicos para ADV L4
        self.df = None
        self.df_L4_ADV_DISC = None  # Discordancias de agujas
        self.df_L4_ADV_MOV = None   # Movimientos de agujas
    
    def find_files(self):
        """Encontrar archivos para análisis ADV de Línea 4"""
        self.csv_files_vid = []  # Para archivos de movimientos (VID)
        self.csv_files_vent = []  # Para archivos de discordancias (vent)
        
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
                            # Verificar si el archivo contiene "VID" (para movimientos)
                            if "VID" in file:
                                try:
                                    self.csv_files_vid.append(os.path.join(folder_path2, file))
                                except ValueError:
                                    print(f"Error al analizar la fecha del archivo: {file}")
                            
                            # Verificar si el archivo contiene "vent" (para discordancias)
                            elif "vent" in file:
                                try:
                                    self.csv_files_vent.append(os.path.join(folder_path2, file))
                                except ValueError:
                                    print(f"Error al analizar la fecha del archivo: {file}")
        
        return len(self.csv_files_vid) + len(self.csv_files_vent)
    
    def read_files(self, progress_callback=None):
        """Leer y procesar archivos para análisis ADV de Línea 4"""
        if progress_callback:
            progress_callback(5, "Procesando archivos para ADV L4...")
        
        # Procesar archivos para movimientos (VID)
        self.df_L4_ADV_MOV = self._process_movement_files(progress_callback)
        
        # Procesar archivos para discordancias (vent)
        self.df_L4_ADV_DISC = self._process_discordance_files(progress_callback)
        
        return self.df_L4_ADV_MOV is not None or self.df_L4_ADV_DISC is not None
    
    def _process_movement_files(self, progress_callback=None):
        """Procesar archivos de movimientos de agujas (VID)"""
        if progress_callback:
            progress_callback(10, "Procesando archivos de movimientos...")
        
        df_list = []
        total_files = len(self.csv_files_vid)
        
        # Función para procesar un solo archivo
        def process_file(index_file_tuple):
            index, csv_file = index_file_tuple
            try:
                # Leer el archivo y filtrar por agujas (AGS)
                df = pd.read_table(csv_file, encoding="Latin-1", sep="|", skiprows=0, header=None, engine='python')
                df = df[df[1].str.contains('AGS', na=False)]
                
                # Filtrar por eventos de posición (normal o reverso)
                df = df[df[2].str.contains('normal|reverso')]
                df = df[df[1].str.contains('posicion')]
                
                if progress_callback and index % 5 == 0:  # Actualizar cada 5 archivos para no saturar la UI
                    progress = 10 + (index / total_files) * 10
                    progress_callback(progress, f"Procesando archivo de movimientos {index+1} de {total_files}")
                
                return df
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"No se pudo leer el archivo {csv_file} debido a un error: {e}")
                return pd.DataFrame()  # Retornar DataFrame vacío en caso de error
        
        # Usar ThreadPoolExecutor para procesar archivos en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
            results = list(executor.map(process_file, enumerate(self.csv_files_vid)))
        
        # Filtrar DataFrames vacíos y concatenar resultados
        df_list = [df for df in results if not df.empty]
        
        if not df_list:
            if progress_callback:
                progress_callback(None, "No se encontraron datos válidos en los archivos de movimientos.")
            return None
        
        # Concatenar resultados
        if progress_callback:
            progress_callback(20, "Procesando datos de movimientos...")
        
        df = pd.concat(df_list, ignore_index=True)
        
        # Procesamiento para extraer información
        df_aux = df[1].str.split(":", expand=True)
        df_aux.columns = ["COL.1", "COL.2"]
        df = df.join(df_aux)
        
        df_aux_2 = df["COL.2"].str.split("_", expand=True)
        df_aux_2.columns = ["COL.3", "COL.4"]
        df = df.join(df_aux_2)
        
        df["COL.1"] = df["COL.1"].str.replace("__", "_")
        df_aux_3 = df["COL.1"].str.split("_", expand=True)
        df_aux_3.columns = ["COL.5", "COL.6", "COL.7", "COL.8", "COL.9"]
        df = df.join(df_aux_3)
        
        # Seleccionar columnas y renombrar
        cols = [0, 4, 8, 10, 11, 7, 2, 3, 5, 6, 1, 9, 12]
        df = df.iloc[:, cols]
        df = df.drop(df.columns[[7, 8, 9, 10, 11, 12]], axis=1)
        
        df.rename(
            columns={0: "Fecha Hora", "COL.1": "Equipo", "COL.5": "Estacion", 
                    "COL.7": "Subsistema", "COL.8": "Numero Equipo", "COL.4": "Atributo", 2: "Estado"},
            inplace=True
        )
        
        # Limpieza y preparación adicional
        df["Numero Equipo"] = df["Numero Equipo"].astype("string")
        df = df.sort_values(["Equipo", "Fecha Hora"])
        df["Fecha Hora"] = pd.to_datetime(df["Fecha Hora"], dayfirst=True)
        df = df.reset_index()
        
        if "index" in df.columns:
            df = df.drop("index", axis=1)
        
        # Eliminar columnas no necesarias
        df = df.drop(["Subsistema", "Numero Equipo", "Atributo"], axis=1)
        
        # Filtrar por horario operativo (6am a 11pm)
        df.set_index('Fecha Hora', inplace=True)
        df = df[(df.index.hour >= 6) & (df.index.hour <= 23)]
        df = df.reset_index()
        
        # Limpiar y procesar equipo
        df["Equipo"] = df["Equipo"].str.replace("TR_", "")
        for i in range(10):
            df["Equipo"] = df["Equipo"].str.replace(f"{i}_", f"{i}")
        
        # Extraer fecha para agrupar por día
        df['Fecha'] = df['Fecha Hora'].dt.date
        
        # Agrupar por equipo y fecha para contar movimientos
        df_mov = df.groupby(['Equipo', 'Fecha']).size().reset_index(name='Count')
        
        # Dividir la columna 'Equipo' para obtener estación
        equipo_split = df_mov['Equipo'].str.split('_', n=1, expand=True)
        df_mov['Estacion'] = equipo_split[0]
        df_mov['Equipo'] = equipo_split[1]
        
        # Crear ID único
        df_mov['Fecha'] = df_mov['Fecha'].astype(str)
        df_mov['Count'] = df_mov['Count'].astype(str)
        df_mov['ID'] = df_mov['Fecha'] + df_mov['Equipo'] + df_mov['Count'] + df_mov['Estacion']
        df_mov['Linea'] = "L4"
        
        return df_mov
    
    def _process_discordance_files(self, progress_callback=None):
        """Procesar archivos de discordancias de agujas (vent)"""
        if progress_callback:
            progress_callback(30, "Procesando archivos de discordancias...")
        
        df_list = []
        total_files = len(self.csv_files_vent)
        
        # Función para procesar un solo archivo
        def process_file(index_file_tuple):
            index, csv_file = index_file_tuple
            try:
                # Leer el archivo y filtrar por agujas (AGS)
                df = pd.read_table(csv_file, encoding="Latin-1", sep="|", skiprows=0, header=None, engine='python')
                df = df[df[1].str.contains('AGS', na=False)]
                
                # Filtrar por discordancias
                df = df[df[2].str.contains('discordancia')]
                
                if progress_callback and index % 5 == 0:  # Actualizar cada 5 archivos para no saturar la UI
                    progress = 30 + (index / total_files) * 10
                    progress_callback(progress, f"Procesando archivo de discordancias {index+1} de {total_files}")
                
                return df
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"No se pudo leer el archivo {csv_file} debido a un error: {e}")
                return pd.DataFrame()  # Retornar DataFrame vacío en caso de error
        
        # Usar ThreadPoolExecutor para procesar archivos en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
            results = list(executor.map(process_file, enumerate(self.csv_files_vent)))
        
        # Filtrar DataFrames vacíos y concatenar resultados
        df_list = [df for df in results if not df.empty]
        
        if not df_list:
            if progress_callback:
                progress_callback(None, "No se encontraron datos válidos en los archivos de discordancias.")
            return None
        
        # Concatenar resultados
        if progress_callback:
            progress_callback(40, "Procesando datos de discordancias...")
        
        df = pd.concat(df_list, ignore_index=True)
        
        # Verificar y eliminar columnas no necesarias
        if 3 in df.columns:
            df = df.drop(columns=[3])
        
        # Renombrar columnas
        df.columns = ["Fecha Hora", "Equipo", "Estado"]
        
        # Procesar columna Equipo
        df_aux = df["Equipo"].str.replace("__", "_")
        df_aux = df_aux.str.split("_", expand=True)
        if not df_aux.empty and len(df_aux.columns) > 4:
            df_aux = df_aux.drop(df_aux.columns[[1, 4]], axis=1)
        df_aux.columns = ["ESTACION", "SISTEMA", "ID_EQUIPO"]
        df = pd.concat([df, df_aux], axis=1)
        
        # Eliminar columna original de Equipo
        df = df.drop("Equipo", axis=1)
        
        # Procesar fecha y hora
        df_aux = df["Fecha Hora"].str.split(" ", expand=True)
        if not df_aux.empty and len(df_aux.columns) > 6:
            df_aux = df_aux.drop(df_aux.columns[[0, 4, 6]], axis=1)
        df_aux.columns = ["MES", "DIA", "HORA", "AÑO"]
        df = pd.concat([df, df_aux], axis=1)
        
        # Eliminar columna original de Fecha Hora
        df = df.drop("Fecha Hora", axis=1)
        
        # Procesar estado
        df_aux = df["Estado"].str.split(":", expand=True)
        df_aux.columns = ["Equipo", "Estado_"]
        df = pd.concat([df, df_aux], axis=1)
        
        # Eliminar columna original de Estado
        df = df.drop("Estado", axis=1)
        
        # Crear columna de fecha formateada
        df["FECHA"] = df.apply(lambda row: f"{row['DIA']}-{row['MES']}-{row['AÑO']}", axis=1)
        df = df.drop("DIA", axis=1)
        df = df.drop("MES", axis=1)
        df = df.drop("AÑO", axis=1)
        
        # Convertir a datetime y formatear
        df["FECHA"] = pd.to_datetime(df["FECHA"], format="%d-%b-%Y", errors='coerce')
        df["FECHA"] = df["FECHA"].dt.strftime("%d-%m-%Y")
        
        # Agregar columna de línea
        df['LINEA'] = 'L4'
        
        # Eliminar columnas no necesarias
        df = df.drop("Equipo", axis=1)
        df = df.drop("SISTEMA", axis=1)
        df = df.drop("Estado_", axis=1)
        
        # Renombrar columnas
        df.columns = ["Estacion", "Equipo", "Hora", "Fecha", "Linea"]
        
        # Crear columna combinada de fecha y hora
        df['Fecha'] = df["Fecha"].astype(str)
        df['Hora'] = df["Hora"].astype(str)
        df["Fecha Hora"] = df["Fecha"] + " " + df["Hora"]
        df = df.drop(columns=['Fecha', 'Hora'])
        
        # Crear columna combinada de equipo y estación
        df["Equipo Estacion"] = df["Equipo"] + "*" + df["Estacion"]
        df = df.drop(columns=['Equipo', 'Estacion'])
        
        # Crear ID único
        df['Fecha Hora'] = df['Fecha Hora'].astype(str)
        df['ID'] = df['Fecha Hora'] + df['Equipo Estacion']
        
        # Convertir y filtrar por hora del día
        df["Fecha Hora"] = pd.to_datetime(df["Fecha Hora"], dayfirst=True, errors='coerce')
        df.set_index('Fecha Hora', inplace=True)
        df = df[(df.index.hour >= 6) & (df.index.hour <= 23)]
        df = df.reset_index()
        
        # Formatear fecha y hora
        df["Fecha Hora"] = df["Fecha Hora"].dt.strftime("%d-%m-%Y %H:%M:%S")
        
        return df
    
    def preprocess_data(self, progress_callback=None):
        """No se requiere preprocesamiento adicional ya que se realizó durante la lectura"""
        if progress_callback:
            progress_callback(50, "Preprocesamiento completado durante la lectura de archivos")
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """No se requiere detección de anomalías ya que se extrajeron directamente de los archivos"""
        if progress_callback:
            progress_callback(70, "Detección de anomalías completada durante la lectura de archivos")
        return True
    
    def prepare_reports(self, progress_callback=None):
        """No se requiere preparación adicional de reportes"""
        if progress_callback:
            progress_callback(80, "Reportes preparados durante la lectura de archivos")
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(85, "Iniciando actualización de reportes...")
            
            # 1. Actualizar reporte de discordancias
            if self.df_L4_ADV_DISC is not None:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L4_ADV_DISC_Mensual.csv')
                if os.path.exists(disc_file_path):
                    if progress_callback:
                        progress_callback(87, "Actualizando reporte de discordancias...")
                    
                    df_L4_ADV_DISC_Mensual = pd.read_csv(disc_file_path)
                    
                    # Concatenar y eliminar duplicados
                    df_L4_ADV_DISC_Mensual = pd.concat([df_L4_ADV_DISC_Mensual, self.df_L4_ADV_DISC], ignore_index=True)
                    df_L4_ADV_DISC_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Convertir a datetime y formatear
                    df_L4_ADV_DISC_Mensual["Fecha Hora"] = pd.to_datetime(df_L4_ADV_DISC_Mensual["Fecha Hora"], 
                                                                      dayfirst=True, errors='coerce')
                    df_L4_ADV_DISC_Mensual["Fecha Hora"] = df_L4_ADV_DISC_Mensual["Fecha Hora"].dt.strftime("%d-%m-%Y %H:%M:%S")
                    
                    # Guardar el resultado actualizado
                    df_L4_ADV_DISC_Mensual.to_csv(disc_file_path, index=False)
                else:
                    # Si el archivo no existe, guardar el nuevo
                    self.df_L4_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # 2. Actualizar reporte de movimientos
            if self.df_L4_ADV_MOV is not None:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L4_ADV_MOV_Mensual.csv')
                if os.path.exists(mov_file_path):
                    if progress_callback:
                        progress_callback(93, "Actualizando reporte de movimientos...")
                    
                    df_L4_ADV_MOV_Mensual = pd.read_csv(mov_file_path)
                    
                    # Concatenar y eliminar duplicados
                    df_L4_ADV_MOV_Mensual = pd.concat([df_L4_ADV_MOV_Mensual, self.df_L4_ADV_MOV], ignore_index=True)
                    df_L4_ADV_MOV_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Guardar el resultado actualizado
                    df_L4_ADV_MOV_Mensual.to_csv(mov_file_path, index=False)
                else:
                    # Si el archivo no existe, guardar el nuevo
                    self.df_L4_ADV_MOV.to_csv(mov_file_path, index=False)
            
            if progress_callback:
                progress_callback(95, "Actualización de reportes completada")
            
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error al actualizar reportes: {str(e)}")
            return False
    
    def save_dataframe(self):
        """Guardar los DataFrames principales"""
        try:
            # Guardar datos de discordancias
            if self.df_L4_ADV_DISC is not None:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L4_ADV_DISC.csv')
                self.df_L4_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # Guardar datos de movimientos
            if self.df_L4_ADV_MOV is not None:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L4_ADV_MOV.csv')
                self.df_L4_ADV_MOV.to_csv(mov_file_path, index=False)
            
            return True
        except Exception as e:
            print(f"Error al guardar DataFrames: {e}")
            return False
    
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos
            if progress_callback:
                progress_callback(0, "Buscando archivos para análisis ADV de Línea 4...")
            num_files = self.find_files()
            if num_files == 0:
                if progress_callback:
                    progress_callback(100, "No se encontraron archivos para procesar")
                return False
            
            # 2. Leer y procesar archivos
            if progress_callback:
                progress_callback(5, f"Procesando {num_files} archivos...")
            if not self.read_files(progress_callback):
                return False
            
            # 3. Preprocesamiento (ya realizado durante la lectura)
            if progress_callback:
                progress_callback(50, "Preprocesamiento ya realizado durante la lectura...")
            self.preprocess_data(progress_callback)
            
            # 4. Detección de anomalías (ya realizada durante la lectura)
            if progress_callback:
                progress_callback(70, "Detección de anomalías ya realizada durante la lectura...")
            self.detect_anomalies(progress_callback)
            
            # 5. Preparación de reportes (ya realizada durante la lectura)
            if progress_callback:
                progress_callback(80, "Preparación de reportes ya realizada durante la lectura...")
            self.prepare_reports(progress_callback)
            
            # 6. Actualización de reportes existentes
            if progress_callback:
                progress_callback(85, "Actualizando reportes existentes...")
            self.update_reports(progress_callback)
            
            # 7. Guardar DataFrames principales
            if progress_callback:
                progress_callback(95, "Guardando DataFrames principales...")
            self.save_dataframe()
            
            if progress_callback:
                progress_callback(100, "Procesamiento ADV Línea 4 completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False