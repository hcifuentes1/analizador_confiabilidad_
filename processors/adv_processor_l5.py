# processors/adv_processor_l5.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class ADVProcessorL5(BaseProcessor):
    """Procesador para datos ADV (Agujas) de la Línea 5"""
    
    def __init__(self):
        super().__init__(line="L5", analysis_type="ADV")
        # Atributos específicos para ADV L5
        self.df = None
        self.df_L5_ADV_DISC = None  # Discordancias de agujas
        self.df_L5_ADV_MOV = None   # Movimientos de agujas
    
    def find_files(self):
        """Encontrar archivos TXT para análisis ADV de Línea 5"""
        self.txt_files = []
        
        # Verificar si la ruta existe
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
            
        # Recorrer carpetas y encontrar archivos TXT
        for folder in os.listdir(self.root_folder_path):
            folder_path = os.path.join(self.root_folder_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.txt'):
                        self.txt_files.append(os.path.join(folder_path, file))
        
        return len(self.txt_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos TXT para análisis ADV de Línea 5"""
        df_list = []
        total_files = len(self.txt_files)
        
        for i, txt in enumerate(self.txt_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(txt)}")
                
                # Leer archivo TXT
                df = pd.read_csv(txt, encoding="Latin-1", sep=";", skiprows=7, header=None, engine='python')
                # Filtrar solo filas con datos de agujas
                df = df[df[5].str.contains('Posicion aguja', na=False)]
                df_list.append(df)
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error en archivo {txt}: {str(e)}")
        
        if df_list:
            self.df = pd.concat(df_list, ignore_index=True)
            del df_list
            return True
        else:
            return False
    
    def preprocess_data(self, progress_callback=None):
        """Realizar el preprocesamiento inicial de los datos"""
        if progress_callback:
            progress_callback(20, "Iniciando preprocesamiento de datos...")
        
        # Crear columna de fecha y hora
        self.df["Fecha Hora"] = self.df[0].astype("string") + " " + self.df[1].astype("string")
        self.df["Fecha Hora"] = pd.to_datetime(self.df["Fecha Hora"], dayfirst=True, errors='coerce')
        
        # Eliminar columnas innecesarias y reordenar
        self.df = self.df.drop(self.df.columns[[0,1,6,7]], axis=1)
        cols = [4,2,0,1,3]
        self.df = self.df.iloc[:,cols]
        
        # Renombrar columnas
        self.df.rename(
            columns={"Fecha Hora": "Fecha Hora", 4: "Equipo", 2: "Estacion", 3: "Subsistema", 5: "Estado"},
            inplace=True,
        )
        
        # Agregar identificador de línea y preparar datos para análisis
        self.df["Linea"] = "L5"
        self.df['Estacion'] = self.df['Estacion'].fillna('NA')
        self.df['Equipo Estacion'] = self.df['Equipo'] + '*' + self.df['Estacion']
        
        # Filtrar solo las agujas acopladas
        self.df = self.df[self.df['Equipo'].str.contains('coplada')]
        self.df["Fecha Hora"] = self.df["Fecha Hora"].dt.strftime('%Y-%m-%d %H:%M:%S')
        
# processors/adv_processor_l5.py (continuación)
        # Crear dataframe para discordancias
        self.df_L5_ADV_DISC = self.df.loc[self.df["Estado"].str.contains('discor', na=False)]
        # Filtrar las filas que contienen "RECONOCIDO" en la columna "Estado"
        self.df_L5_ADV_DISC = self.df_L5_ADV_DISC[~self.df_L5_ADV_DISC['Estado'].str.contains('RECONOCIDO')]
        # Comprobar si la columna 'index' existe antes de intentar eliminarla
        if 'index' in self.df_L5_ADV_DISC.columns:
            self.df_L5_ADV_DISC = self.df_L5_ADV_DISC.drop(columns=['index'])
        self.df_L5_ADV_DISC = self.df_L5_ADV_DISC.drop(columns=['Subsistema'])
        self.df_L5_ADV_DISC['Fecha Hora'] = self.df_L5_ADV_DISC['Fecha Hora'].astype(str)
        self.df_L5_ADV_DISC['ID'] = self.df_L5_ADV_DISC['Fecha Hora'] + self.df_L5_ADV_DISC['Equipo Estacion']
        
        # Crear dataframe para movimientos
        # Convertir la columna 'Fecha Hora' a formato de fecha y hora
        self.df['Fecha Hora'] = pd.to_datetime(self.df['Fecha Hora'], errors='coerce')
        # Extraer el año, mes y día de la columna 'Fecha Hora'
        self.df['Fecha'] = self.df['Fecha Hora'].dt.date
        # Rellenar los valores nulos en la columna 'Estacion' con 'NA'
        self.df['Estacion'] = self.df['Estacion'].fillna('NA')
        # Filtrar y agrupar para movimientos
        self.df_L5_ADV_MOV = self.df[self.df['Estado'].str.contains('Posicion')]
        self.df_L5_ADV_MOV = self.df[self.df['Equipo'].str.contains('coplada')]
        
        # Agrupar por equipo y fecha, contando ocurrencias
        self.df_L5_ADV_MOV = self.df_L5_ADV_MOV.groupby(['Equipo Estacion', 'Fecha']).size().reset_index(name='Count')
        self.df_L5_ADV_MOV[['Equipo', 'Estacion']] = self.df_L5_ADV_MOV['Equipo Estacion'].str.split('*', expand=True)
        self.df_L5_ADV_MOV['Estacion'] = self.df_L5_ADV_MOV['Estacion'].fillna('NA')
        
        self.df_L5_ADV_MOV = self.df_L5_ADV_MOV.drop(columns=['Equipo Estacion'])
        self.df_L5_ADV_MOV['Estacion'] = self.df_L5_ADV_MOV['Estacion'].fillna('NA')
        self.df_L5_ADV_MOV['Fecha'] = self.df_L5_ADV_MOV['Fecha'].astype(str)
        self.df_L5_ADV_MOV['Count'] = self.df_L5_ADV_MOV['Count'].astype(str)
        self.df_L5_ADV_MOV['ID'] = self.df_L5_ADV_MOV['Fecha'] + self.df_L5_ADV_MOV['Equipo'] + self.df_L5_ADV_MOV['Count'] + self.df_L5_ADV_MOV['Estacion']
        
        if progress_callback:
            progress_callback(40, "Preprocesamiento completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """En el caso de ADV, las anomalías ya están identificadas en la fase de preprocesamiento"""
        if progress_callback:
            progress_callback(70, "Procesamiento de anomalías completado")
        return True
    
    def prepare_reports(self, progress_callback=None):
        """Preparación adicional de reportes no es necesaria para ADV"""
        if progress_callback:
            progress_callback(80, "Reportes preparados")
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(85, "Iniciando actualización de reportes...")
            
            # 1. Actualizar reporte de discordancias
            disc_file_path = os.path.join(self.output_folder_path, 'df_L5_ADV_DISC_Mensual.csv')
            if os.path.exists(disc_file_path):
                df_L5_ADV_DISC_Mensual = pd.read_csv(disc_file_path)
                df_L5_ADV_DISC_Mensual['Estacion'] = df_L5_ADV_DISC_Mensual['Estacion'].fillna('NA')
                
                # Concatenar y eliminar duplicados
                df_L5_ADV_DISC_Mensual = pd.concat([df_L5_ADV_DISC_Mensual, self.df_L5_ADV_DISC], ignore_index=True)
                df_L5_ADV_DISC_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                df_L5_ADV_DISC_Mensual['Estacion'] = df_L5_ADV_DISC_Mensual['Estacion'].fillna('NA')
                
                # Guardar el resultado actualizado
                df_L5_ADV_DISC_Mensual.to_csv(disc_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L5_ADV_DISC.to_csv(disc_file_path, index=False)
            
            # 2. Actualizar reporte de movimientos
            mov_file_path = os.path.join(self.output_folder_path, 'df_L5_ADV_MOV_Mensual.csv')
            if os.path.exists(mov_file_path):
                df_L5_ADV_MOV_Mensual = pd.read_csv(mov_file_path)
                df_L5_ADV_MOV_Mensual['Estacion'] = df_L5_ADV_MOV_Mensual['Estacion'].fillna('NA')
                
                # Concatenar y eliminar duplicados
                df_L5_ADV_MOV_Mensual = pd.concat([df_L5_ADV_MOV_Mensual, self.df_L5_ADV_MOV], ignore_index=True)
                df_L5_ADV_MOV_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                df_L5_ADV_MOV_Mensual['Estacion'] = df_L5_ADV_MOV_Mensual['Estacion'].fillna('NA')
                
                # Guardar el resultado actualizado
                df_L5_ADV_MOV_Mensual.to_csv(mov_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L5_ADV_MOV.to_csv(mov_file_path, index=False)
            
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
            # Guardar el DataFrame principal
            main_file_path = os.path.join(self.output_folder_path, 'df_L5_ADV.csv')
            self.df.to_csv(main_file_path, index=False)
            
            return True
        except Exception as e:
            return False