# processors/adv_processor_l1.py
import pandas as pd
import numpy as np
import os
import zipfile
import io
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class ADVProcessorL1(BaseProcessor):
    """Procesador para datos ADV (Agujas) de la Línea 1"""
    
    def __init__(self):
        super().__init__(line="L1", analysis_type="ADV")
        # Atributos específicos para ADV L1
        self.df_L1_ADV_DISC = None  # Discordancias de agujas
        self.df_L1_ADV_MOV = None   # Movimientos de agujas
        self.zip_files_alarmlist = []
        self.zip_files_s2k = []
        self.extracted_files = []
    
    def find_files(self):
        """Encontrar archivos ZIP para análisis ADV de Línea 1"""
        self.zip_files_alarmlist = []
        self.zip_files_s2k = []
        
        # Buscar archivos AlarmList (discordancias)
        alarmlist_path = os.path.join(self.root_folder_path, 'CBI Alarmlist')
        if os.path.exists(alarmlist_path):
            for root, _, files in os.walk(alarmlist_path):
                for file in files:
                    if file.endswith('.zip') and 'CBI_1_AlarmList' in file:
                        self.zip_files_alarmlist.append(os.path.join(root, file))
        
        # Buscar archivos S2K (movimientos)
        s2k_path = os.path.join(self.root_folder_path, 'S2K')
        if os.path.exists(s2k_path):
            for root, _, files in os.walk(s2k_path):
                for file in files:
                    if file.endswith('.zip'):
                        self.zip_files_s2k.append(os.path.join(root, file))
        
        return len(self.zip_files_alarmlist) + len(self.zip_files_s2k)
    
    def read_files(self, progress_callback=None):
        """Leer archivos ZIP para análisis ADV de Línea 1"""
        # Procesar archivos AlarmList (discordancias)
        if progress_callback:
            progress_callback(5, "Procesando archivos de discordancias...")
        
        discordancias_dfs = []
        for i, zip_file in enumerate(self.zip_files_alarmlist):
            try:
                discordancias_dfs.extend(self.extract_filtered_rows_from_alarmlist_zip(zip_file))
                if progress_callback:
                    progress = 5 + (i / len(self.zip_files_alarmlist)) * 10
                    progress_callback(progress, f"Procesando archivo {i+1} de {len(self.zip_files_alarmlist)}: {os.path.basename(zip_file)}")
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error al procesar {os.path.basename(zip_file)}: {str(e)}")
        
        # Procesar archivos S2K (movimientos)
        if progress_callback:
            progress_callback(15, "Procesando archivos de movimientos...")
        
        movimientos_dfs = []
        for i, zip_file in enumerate(self.zip_files_s2k):
            try:
                # Extraer y procesar archivos S2K
                extracted_files = self.extract_zip_file(zip_file)
                
                for file_path in extracted_files:
                    if file_path.endswith('.csv'):
                        df = self.read_and_clean_csv(file_path)
                        if not df.empty:
                            movimientos_dfs.append(df)
                
                if progress_callback:
                    progress = 15 + (i / len(self.zip_files_s2k)) * 10
                    progress_callback(progress, f"Procesando archivo {i+1} de {len(self.zip_files_s2k)}: {os.path.basename(zip_file)}")
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error al procesar {os.path.basename(zip_file)}: {str(e)}")
        
        # Combinar resultados
        if discordancias_dfs:
            self.df_L1_ADV_DISC = pd.concat(discordancias_dfs, ignore_index=True)
            # Limpiar
            self.df_L1_ADV_DISC = self.df_L1_ADV_DISC.copy()
            self.df_L1_ADV_DISC['Estacion'] = self.df_L1_ADV_DISC['Equipo'].str[-2:]
            self.df_L1_ADV_DISC["Linea"] = "L1"
        
        if movimientos_dfs:
            combined_df = pd.concat(movimientos_dfs, ignore_index=True)
            # Extraer el estado (últimos 7 caracteres)
            combined_df['Estado'] = combined_df['Estado'].apply(lambda x: x[-7:] if isinstance(x, str) else x)
            
            # Convertir la columna 'Fecha Hora' a formato de fecha y hora
            combined_df['Fecha Hora'] = pd.to_datetime(combined_df['Fecha Hora'], errors='coerce')
            # Extraer el año, mes y día de la columna 'Fecha Hora'
            combined_df['Fecha'] = combined_df['Fecha Hora'].dt.date
            
            # Agrupar y contar
            self.df_L1_ADV_MOV = combined_df.groupby(['Equipo', 'Fecha']).size().reset_index(name='Count')
            self.df_L1_ADV_MOV['Estacion'] = self.df_L1_ADV_MOV['Equipo'].apply(lambda x: x[-2:] if isinstance(x, str) else x)
        
        # Limpiar archivos extraídos temporales
        for file_path in self.extracted_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        self.extracted_files = []
        
        return (self.df_L1_ADV_DISC is not None) or (self.df_L1_ADV_MOV is not None)
    
    def extract_filtered_rows_from_alarmlist_zip(self, archivo_zip):
        """Extraer y filtrar datos de archivos Excel en un ZIP de AlarmList"""
        filas_filtradas = []
        try:
            with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
                for nombre_archivo in zip_ref.namelist():
                    if nombre_archivo.endswith('.xls'):
                        try:
                            contenido = zip_ref.read(nombre_archivo)
                            with io.BytesIO(contenido) as f:
                                df = pd.read_excel(f)
                            # Filtrar columnas
                            df = df.iloc[:, [0, 1, 4]]
                            # Renombrar columnas
                            df.columns = ['Fecha Hora', 'Equipo', 'Estado']
                            # Filtrar filas
                            df = df[(df['Equipo'].str.contains('AG', na=False)) & 
                                    (df['Estado'].str.contains('DISCREP', na=False))]
                            filas_filtradas.append(df)
                        except Exception as e:
                            print(f"Error al procesar {nombre_archivo}: {e}")
        except Exception as e:
            print(f"Error al abrir ZIP {archivo_zip}: {e}")
        
        return filas_filtradas
    
    def extract_zip_file(self, ruta_zip):
        """Extraer archivos de un archivo ZIP y devolver una lista de rutas de archivos extraídos."""
        rutas_archivos = []
        try:
            # Crear un directorio temporal único para este ZIP
            temp_dir = os.path.join(os.getcwd(), f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            os.makedirs(temp_dir, exist_ok=True)
            
            with zipfile.ZipFile(ruta_zip, 'r') as archivo_zip:
                for archivo in archivo_zip.namelist():
                    # Extraer solo archivos CSV
                    if archivo.endswith('.csv'):
                        archivo_zip.extract(archivo, temp_dir)
                        ruta_completa = os.path.join(temp_dir, archivo)
                        rutas_archivos.append(ruta_completa)
                        self.extracted_files.append(ruta_completa)
        except Exception as e:
            print(f"Error al extraer {ruta_zip}: {e}")
        
        return rutas_archivos

    def read_and_clean_csv(self, file_path):
        """Leer y limpiar un archivo CSV, eliminando caracteres problemáticos y seleccionando columnas específicas."""
        try:
            with open(file_path, 'r', encoding='Latin-1') as file:
                content = file.read()
            
            # Reemplazar caracteres problemáticos
            content = content.replace('\x00', '')

            # Usar StringIO para leer el contenido limpiado en un DataFrame
            df = pd.read_csv(io.StringIO(content), sep=',', header=None, encoding='Latin-1', engine='python')
            
            # Seleccionar y renombrar columnas según sea necesario
            df = df[[0, 2, 9]]
            df.columns = ['Equipo', 'Fecha Hora', 'Estado']
            df = df[(df['Equipo'].str.contains('AG_', na=False)) & 
                    (df['Estado'].str.contains('en posición', na=False)) &
                    (~df['Estado'].str.contains('libre', na=False)) & 
                    (~df['Estado'].str.contains('ocupada', na=False))]
            
            return df
        except Exception as e:
            print(f"Error al procesar el archivo '{file_path}': {e}")
            return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error
    
    def preprocess_data(self, progress_callback=None):
        """Realizar el preprocesamiento inicial de los datos"""
        if progress_callback:
            progress_callback(25, "Iniciando preprocesamiento de datos...")
        
        # Preprocesar datos de discordancias
        if self.df_L1_ADV_DISC is not None:
            if progress_callback:
                progress_callback(30, "Procesando datos de discordancias...")
            
            # Convertir la columna 'Fecha Hora' a formato datetime
            self.df_L1_ADV_DISC["Fecha Hora"] = pd.to_datetime(self.df_L1_ADV_DISC["Fecha Hora"], dayfirst=True, errors='coerce')
            
            # Ordenar por equipo y fecha/hora
            self.df_L1_ADV_DISC = self.df_L1_ADV_DISC.sort_values(["Equipo", "Fecha Hora"])
            self.df_L1_ADV_DISC = self.df_L1_ADV_DISC.reset_index()
            
            # Filtrar por hora del día (horario operativo)
            self.df_L1_ADV_DISC.set_index('Fecha Hora', inplace=True)
            self.df_L1_ADV_DISC = self.df_L1_ADV_DISC[(self.df_L1_ADV_DISC.index.hour >= 6) & (self.df_L1_ADV_DISC.index.hour <= 23)]
            self.df_L1_ADV_DISC = self.df_L1_ADV_DISC.reset_index()
            
            # Eliminar columna 'Estado' ya que no es necesaria para discordancias
            if 'Estado' in self.df_L1_ADV_DISC.columns:
                self.df_L1_ADV_DISC = self.df_L1_ADV_DISC.drop(columns=['Estado'])
            
            # Eliminar columna 'index' si existe
            if 'index' in self.df_L1_ADV_DISC.columns:
                self.df_L1_ADV_DISC = self.df_L1_ADV_DISC.drop(columns=['index'])
        
        # Preprocesar datos de movimientos
        if self.df_L1_ADV_MOV is not None:
            if progress_callback:
                progress_callback(35, "Procesando datos de movimientos...")
            
            # Convertir columnas a string para crear ID único
            self.df_L1_ADV_MOV['Fecha'] = self.df_L1_ADV_MOV['Fecha'].astype(str)
            self.df_L1_ADV_MOV['Count'] = self.df_L1_ADV_MOV['Count'].astype(str)
            self.df_L1_ADV_MOV['Linea'] = "L1"
        
        if progress_callback:
            progress_callback(40, "Preprocesamiento completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """Para ADV, solo necesitamos crear IDs únicos para discordancias y movimientos"""
        if progress_callback:
            progress_callback(70, "Preparando identificadores únicos...")
        
        # Crear ID único para discordancias
        if self.df_L1_ADV_DISC is not None:
            self.df_L1_ADV_DISC['Fecha Hora'] = self.df_L1_ADV_DISC['Fecha Hora'].astype(str)
            self.df_L1_ADV_DISC['ID'] = self.df_L1_ADV_DISC['Fecha Hora'] + self.df_L1_ADV_DISC['Equipo']
        
        # Crear ID único para movimientos
        if self.df_L1_ADV_MOV is not None:
            self.df_L1_ADV_MOV['ID'] = self.df_L1_ADV_MOV['Fecha'] + self.df_L1_ADV_MOV['Equipo'] + self.df_L1_ADV_MOV['Count'] + self.df_L1_ADV_MOV['Estacion']
        
        if progress_callback:
            progress_callback(75, "Identificadores únicos creados")
        
        return True
    
    def prepare_reports(self, progress_callback=None):
        """No necesitamos preparación adicional para ADV más allá de lo que ya se hizo"""
        if progress_callback:
            progress_callback(80, "Preparación de reportes completada")
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(85, "Iniciando actualización de reportes...")
            
            # 1. Actualizar reporte de discordancias
            if self.df_L1_ADV_DISC is not None:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L1_ADV_DISC_Mensual.csv')
                if os.path.exists(disc_file_path):
                    if progress_callback:
                        progress_callback(87, "Actualizando reporte de discordancias...")
                    
                    df_L1_ADV_DISC_Mensual = pd.read_csv(disc_file_path)
                    
                    # Concatenar y eliminar duplicados
                    df_L1_ADV_DISC_Mensual = pd.concat([df_L1_ADV_DISC_Mensual, self.df_L1_ADV_DISC], ignore_index=True)
                    df_L1_ADV_DISC_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Guardar el resultado actualizado
                    df_L1_ADV_DISC_Mensual.to_csv(disc_file_path, index=False)
                else:
                    # Si el archivo no existe, guardar el nuevo
                    self.df_L1_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # 2. Actualizar reporte de movimientos
            if self.df_L1_ADV_MOV is not None:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L1_ADV_MOV_Mensual.csv')
                if os.path.exists(mov_file_path):
                    if progress_callback:
                        progress_callback(93, "Actualizando reporte de movimientos...")
                    
                    df_L1_ADV_MOV_Mensual = pd.read_csv(mov_file_path)
                    
                    # Concatenar y eliminar duplicados
                    df_L1_ADV_MOV_Mensual = pd.concat([df_L1_ADV_MOV_Mensual, self.df_L1_ADV_MOV], ignore_index=True)
                    df_L1_ADV_MOV_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    
                    # Guardar el resultado actualizado
                    df_L1_ADV_MOV_Mensual.to_csv(mov_file_path, index=False)
                else:
                    # Si el archivo no existe, guardar el nuevo
                    self.df_L1_ADV_MOV.to_csv(mov_file_path, index=False)
            
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
            if self.df_L1_ADV_DISC is not None:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L1_ADV_DISC.csv')
                self.df_L1_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # Guardar datos de movimientos
            if self.df_L1_ADV_MOV is not None:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L1_ADV_MOV.csv')
                self.df_L1_ADV_MOV.to_csv(mov_file_path, index=False)
            
            return True
        except Exception as e:
            print(f"Error al guardar DataFrames: {e}")
            return False