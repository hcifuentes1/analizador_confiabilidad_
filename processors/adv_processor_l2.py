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
        
        # Atributo para el tipo de datos
        self.data_type = "Sacem"  # Valor por defecto
    
    def set_data_type(self, data_type):
        """Establecer tipo de datos (Sacem o SCADA)"""
        self.data_type = data_type
    
    def find_files(self):
        """Encontrar archivos CSV/Excel para análisis ADV de Línea 2"""
        self.data_files = []
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Recorrer carpetas y encontrar archivos CSV o Excel
        for root, dirs, files in os.walk(self.root_folder_path):
            for file in files:
                # Filtrar según el tipo de datos
                if self.data_type == "Sacem":
                    # Buscar archivos de Sacem relacionados con agujas
                    if file.endswith(('.xlsx', '.xls', '.csv')) and ('AGS' in file or 'adv' in file.lower() or 'aguja' in file.lower()):
                        self.data_files.append(os.path.join(root, file))
                elif self.data_type == "SCADA":
                    # En el futuro, aquí se implementará la lógica para archivos SCADA
                    # Por ahora, solo registramos que está en desarrollo
                    continue
        
        return len(self.data_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos CSV/Excel para análisis ADV de Línea 2"""
        # Listas para almacenar datos de movimientos y discordancias
        mov_data_frames = []
        disc_data_frames = []
        total_files = len(self.data_files)
        
        for i, file_path in enumerate(self.data_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(file_path)}")
                
                # Leer archivo según su extensión
                if file_path.lower().endswith('.csv'):
                    # Probar diferentes codificaciones y separadores comunes para CSV
                    try:
                        # Primero intentar con separador punto y coma (común en configuraciones europeas)
                        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                    except:
                        try:
                            # Luego intentar con separador coma (estándar)
                            df = pd.read_csv(file_path, sep=',', encoding='utf-8')
                        except:
                            try:
                                # Intentar con codificación Latin-1
                                df = pd.read_csv(file_path, sep=';', encoding='latin1')
                            except:
                                # Último intento con coma y Latin-1
                                df = pd.read_csv(file_path, sep=',', encoding='latin1')
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
                
                # Determinar si el archivo contiene datos de movimientos o discordancias
                is_movement_file = False
                is_discordance_file = False
                
                # Comprobar el contenido del archivo para clasificarlo
                if 'FECHA' in df.columns and 'HORA' in df.columns:
                    # Verificar columnas con nombres que sugieran agujas
                    ags_cols = [col for col in df.columns if (
                        col.startswith('AGS') or 
                        'AGUJA' in col.upper() or 
                        'ADV' in col.upper()
                    )]
                    
                    if ags_cols:
                        # Si no hay columnas con 'DISCOR' o 'FALLO' en el nombre, asumimos que son movimientos
                        if not any('DISCOR' in col.upper() or 'FALLO' in col.upper() for col in df.columns):
                            is_movement_file = True
                        else:
                            is_discordance_file = True
                
                # Procesar movimientos
                if is_movement_file:
                    # Tomar las columnas necesarias
                    ags_cols = [col for col in df.columns if (
                        col.startswith('AGS') or 
                        'AGUJA' in col.upper() or 
                        'ADV' in col.upper()
                    )]
                    
                    cols_to_keep = ['ciclo', 'FECHA', 'HORA'] + ags_cols
                    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
                    
                    if len(cols_to_keep) > 3:  # Asegurarnos de que hay al menos una columna de aguja
                        df_subset = df[cols_to_keep]
                        
                        # Crear una versión "derretida" (melted) del dataframe
                        id_vars = ['ciclo', 'FECHA', 'HORA']
                        id_vars = [col for col in id_vars if col in df_subset.columns]
                        
                        df_melted = pd.melt(
                            df_subset, 
                            id_vars=id_vars, 
                            value_vars=[col for col in cols_to_keep if col not in id_vars],
                            var_name='Equipo',
                            value_name='Movimiento'
                        )
                        
                        # Añadir información básica
                        df_melted['Estacion'] = df_melted['Equipo'].str.extract(r'AGS\s*(\w+)')
                        df_melted['Fecha Hora'] = pd.to_datetime(
                            df_melted['FECHA'].astype(str) + ' ' + df_melted['HORA'].astype(str),
                            errors='coerce'
                        )
                        
                        # Solo mantener registros cuando hubo un movimiento (valor = 1)
                        df_melted = df_melted[df_melted['Movimiento'] == 1]
                        
                        # Agregar a la lista de movimientos
                        mov_data_frames.append(df_melted)
                
                # Procesar discordancias
                elif is_discordance_file:
                    # Procesar de manera similar a los movimientos, pero con criterio de discordancia
                    ags_cols = [col for col in df.columns if (
                        (col.startswith('AGS') or 'AGUJA' in col.upper() or 'ADV' in col.upper()) and
                        ('DISCOR' in col.upper() or 'FALLO' in col.upper())
                    )]
                    
                    cols_to_keep = ['ciclo', 'FECHA', 'HORA'] + ags_cols
                    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
                    
                    if len(cols_to_keep) > 3:  # Asegurarnos de que hay al menos una columna de discordancia
                        df_subset = df[cols_to_keep]
                        
                        # Crear una versión "derretida" (melted) del dataframe
                        id_vars = ['ciclo', 'FECHA', 'HORA']
                        id_vars = [col for col in id_vars if col in df_subset.columns]
                        
                        df_melted = pd.melt(
                            df_subset, 
                            id_vars=id_vars, 
                            value_vars=[col for col in cols_to_keep if col not in id_vars],
                            var_name='Equipo',
                            value_name='Discordancia'
                        )
                        
                        # Añadir información básica
                        df_melted['Estacion'] = df_melted['Equipo'].str.extract(r'AGS\s*(\w+)')
                        df_melted['Fecha Hora'] = pd.to_datetime(
                            df_melted['FECHA'].astype(str) + ' ' + df_melted['HORA'].astype(str),
                            errors='coerce'
                        )
                        
                        # Solo mantener registros cuando hubo una discordancia (valor = 1)
                        df_melted = df_melted[df_melted['Discordancia'] == 1]
                        
                        # Agregar a la lista de discordancias
                        disc_data_frames.append(df_melted)
            
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error en archivo {file_path}: {str(e)}")
        
        # Procesar datos de movimientos
        if mov_data_frames:
            if progress_callback:
                progress_callback(20, "Procesando datos de movimientos...")
            
            # Combinar todos los data frames de movimientos
            mov_df = pd.concat(mov_data_frames, ignore_index=True)
            
            # Eliminar duplicados si los hay
            mov_df = mov_df.drop_duplicates()
            
            # Filtrar por hora del día (operación del metro, de 6am a 11pm)
            mov_df = mov_df[
                (mov_df['Fecha Hora'].dt.hour >= 6) & 
                (mov_df['Fecha Hora'].dt.hour <= 23)
            ]
            
            # Extraer solo la fecha (sin hora)
            mov_df['Fecha'] = mov_df['Fecha Hora'].dt.date
            
            # Agrupar por equipo y fecha para contar movimientos
            self.df_L2_ADV_MOV = mov_df.groupby(['Equipo', 'Estacion', 'Fecha']).size().reset_index(name='Count')
            
            # Crear ID único para cada registro
            self.df_L2_ADV_MOV['Fecha'] = self.df_L2_ADV_MOV['Fecha'].astype(str)
            self.df_L2_ADV_MOV['Count'] = self.df_L2_ADV_MOV['Count'].astype(str)
            self.df_L2_ADV_MOV['Equipo'] = self.df_L2_ADV_MOV['Equipo'].astype(str)
            self.df_L2_ADV_MOV['Estacion'] = self.df_L2_ADV_MOV['Estacion'].fillna('NA').astype(str)
            self.df_L2_ADV_MOV['ID'] = self.df_L2_ADV_MOV['Fecha'] + self.df_L2_ADV_MOV['Equipo'] + self.df_L2_ADV_MOV['Count'] + self.df_L2_ADV_MOV['Estacion']
            
            # Agregar columna de línea
            self.df_L2_ADV_MOV['Linea'] = 'L2'
            
            # Crear columna combinada de equipo y estación
            self.df_L2_ADV_MOV['Equipo Estacion'] = self.df_L2_ADV_MOV['Equipo'] + '*' + self.df_L2_ADV_MOV['Estacion']
        
        # Procesar datos de discordancias
        if disc_data_frames:
            if progress_callback:
                progress_callback(30, "Procesando datos de discordancias...")
            
            # Combinar todos los data frames de discordancias
            disc_df = pd.concat(disc_data_frames, ignore_index=True)
            
            # Eliminar duplicados si los hay
            disc_df = disc_df.drop_duplicates()
            
            # Filtrar por hora del día (operación del metro, de 6am a 11pm)
            disc_df = disc_df[
                (disc_df['Fecha Hora'].dt.hour >= 6) & 
                (disc_df['Fecha Hora'].dt.hour <= 23)
            ]
            
            # Preparar DataFrame de discordancias
            self.df_L2_ADV_DISC = disc_df[['Equipo', 'Estacion', 'Fecha Hora']].copy()
            
            # Crear columna combinada de equipo y estación
            self.df_L2_ADV_DISC['Estacion'] = self.df_L2_ADV_DISC['Estacion'].fillna('NA').astype(str)
            self.df_L2_ADV_DISC['Equipo Estacion'] = self.df_L2_ADV_DISC['Equipo'] + '*' + self.df_L2_ADV_DISC['Estacion']
            
            # Agregar columna de línea
            self.df_L2_ADV_DISC['Linea'] = 'L2'
            
            # Formatear fecha y hora
            self.df_L2_ADV_DISC['Fecha Hora'] = self.df_L2_ADV_DISC['Fecha Hora'].dt.strftime("%d-%m-%Y %H:%M:%S")
            
            # Crear ID único para cada registro
            self.df_L2_ADV_DISC['ID'] = self.df_L2_ADV_DISC['Fecha Hora'] + self.df_L2_ADV_DISC['Equipo Estacion']
        
        return (self.df_L2_ADV_MOV is not None) or (self.df_L2_ADV_DISC is not None)
    
    def preprocess_data(self, progress_callback=None):
        """No se requiere preprocesamiento adicional ya que se realizó durante la lectura"""
        if progress_callback:
            progress_callback(50, "Preprocesamiento completado durante la lectura de archivos")
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """No se requiere detección de anomalías ya que se extraen directamente de los archivos"""
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
            if self.df_L2_ADV_DISC is not None:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC_Mensual.csv')
                if os.path.exists(disc_file_path):
                    if progress_callback:
                        progress_callback(87, "Actualizando reporte de discordancias...")
                    
                    df_L2_ADV_DISC_Mensual = pd.read_csv(disc_file_path)
                    df_L2_ADV_DISC_Mensual['Estacion'] = df_L2_ADV_DISC_Mensual['Estacion'].fillna('NA')
                    
                    # Concatenar y eliminar duplicados
                    df_L2_ADV_DISC_Mensual = pd.concat([df_L2_ADV_DISC_Mensual, self.df_L2_ADV_DISC], ignore_index=True)
                    df_L2_ADV_DISC_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    df_L2_ADV_DISC_Mensual['Estacion'] = df_L2_ADV_DISC_Mensual['Estacion'].fillna('NA')
                    
                    # Guardar el resultado actualizado
                    df_L2_ADV_DISC_Mensual.to_csv(disc_file_path, index=False)
                else:
                    # Si el archivo no existe, guardar el nuevo
                    self.df_L2_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # 2. Actualizar reporte de movimientos
            if self.df_L2_ADV_MOV is not None:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV_Mensual.csv')
                if os.path.exists(mov_file_path):
                    if progress_callback:
                        progress_callback(93, "Actualizando reporte de movimientos...")
                    
                    df_L2_ADV_MOV_Mensual = pd.read_csv(mov_file_path)
                    df_L2_ADV_MOV_Mensual['Estacion'] = df_L2_ADV_MOV_Mensual['Estacion'].fillna('NA')
                    
                    # Concatenar y eliminar duplicados
                    df_L2_ADV_MOV_Mensual = pd.concat([df_L2_ADV_MOV_Mensual, self.df_L2_ADV_MOV], ignore_index=True)
                    df_L2_ADV_MOV_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                    df_L2_ADV_MOV_Mensual['Estacion'] = df_L2_ADV_MOV_Mensual['Estacion'].fillna('NA')
                    
                    # Guardar el resultado actualizado
                    df_L2_ADV_MOV_Mensual.to_csv(mov_file_path, index=False)
                else:
                    # Si el archivo no existe, guardar el nuevo
                    self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
            
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
            if self.df_L2_ADV_DISC is not None:
                disc_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_DISC.csv')
                self.df_L2_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # Guardar datos de movimientos
            if self.df_L2_ADV_MOV is not None:
                mov_file_path = os.path.join(self.output_folder_path, 'df_L2_ADV_MOV.csv')
                self.df_L2_ADV_MOV.to_csv(mov_file_path, index=False)
            
            return True
        except Exception as e:
            print(f"Error al guardar DataFrames: {e}")
            return False
    
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos
            if progress_callback:
                progress_callback(0, "Buscando archivos para análisis ADV de Línea 2...")
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
                progress_callback(100, "Procesamiento ADV Línea 2 completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False