# processors/cdv_processor_l1.py
import pandas as pd
import numpy as np
import os
import zipfile
import io
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class CDVProcessorL1(BaseProcessor):
    """Procesador para datos CDV de la Línea 1"""
    
    def __init__(self):
        super().__init__(line="L1", analysis_type="CDV")
        # Atributos específicos para CDV L1
        self.df = None
        self.df_L1_2 = None
        self.df_L1_FO = None
        self.df_L1_FL = None
        self.df_L1_OCUP = None
        
        # Factores de umbral para detección de anomalías
        self.f_oc_1 = 0.1
        self.f_lb_2 = 0.05
    
    def find_files(self):
        """Encontrar archivos ZIP para análisis CDV de Línea 1"""
        self.zip_files = []
        
        # Verificar si la ruta existe
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(f"La ruta {self.root_folder_path} no existe")
        
        # Recorrer el directorio SMIO_CBI para buscar archivos ZIP
        smio_path = os.path.join(self.root_folder_path, 'SMIO_CBI')
        if os.path.exists(smio_path):
            for root, _, files in os.walk(smio_path):
                for file in files:
                    if file.endswith('.zip') and 'SMIO_CBI' in file:
                        self.zip_files.append(os.path.join(root, file))
        
        return len(self.zip_files)
    
    def read_files(self, progress_callback=None):
        """Leer archivos ZIP para análisis CDV de Línea 1"""
        data = []
        total_files = len(self.zip_files)
        
        for i, zip_path in enumerate(self.zip_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(zip_path)}")
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for csv_filename in zip_ref.namelist():
                        if csv_filename.endswith('.csv'):
                            try:
                                with zip_ref.open(csv_filename) as file:
                                    df = pd.read_csv(file)
                                    
                                    # Tomar la segunda columna (Fecha Hora)
                                    fecha_hora = df.iloc[:, 1]
                                    
                                    # Iterar sobre cada columna (ignorando las primeras dos)
                                    for col_name in df.columns[2:]:
                                        # Conservar las filas donde haya un valor no nulo
                                        non_empty_rows = df[df[col_name].notna()]
                                        
                                        # Generar el DataFrame resultante
                                        new_df = pd.DataFrame({
                                            'Fecha Hora': fecha_hora[non_empty_rows.index],
                                            'Estado': non_empty_rows[col_name],
                                            'Equipo': col_name
                                        })
                                        
                                        # Convertir 'Fecha Hora' a formato datetime
                                        new_df["Fecha Hora"] = new_df["Fecha Hora"].astype("datetime64[ns]")
                                        
                                        # Extraer los últimos 9 caracteres de 'Equipo'
                                        new_df["Equipo"] = new_df["Equipo"].str.slice(start=-9)
                                        
                                        # Ordenar y eliminar duplicados para mantener solo los eventos únicos
                                        new_df = new_df.sort_values(by=['Fecha Hora', 'Equipo'])
                                        new_df["Estado"] = new_df["Estado"].astype("int64")
                                        new_df = new_df[new_df['Estado'].isin([1, 0])]
                                        
                                        # Eliminar eventos duplicados consecutivos
                                        new_df["Diff_Aux"] = new_df["Estado"].diff().astype(str)
                                        new_df = new_df[~new_df["Diff_Aux"].str.contains("0.0")]
                                        new_df = new_df.drop("Diff_Aux", axis=1)
                                        
                                        # Agregar el DataFrame a la lista
                                        data.append(new_df)
                            except Exception as e:
                                if progress_callback:
                                    progress_callback(None, f"Error al procesar {csv_filename} en {zip_path}: {str(e)}")
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error al abrir {zip_path}: {str(e)}")
        
        if data:
            self.df = pd.concat(data, ignore_index=True)
            return True
        else:
            return False
    
    def preprocess_data(self, progress_callback=None):
        """Preprocesar datos para análisis CDV de Línea 1"""
        if progress_callback:
            progress_callback(20, "Preprocesando datos...")
        
        # Convertir a string para crear ID único
        self.df["Fecha Hora"] = self.df["Fecha Hora"].astype(str)
        self.df["Estado"] = self.df["Estado"].astype(str)
        self.df['ID'] = self.df['Fecha Hora'] + "*" + self.df['Estado'] + "*" + self.df['Equipo']
        
        # Convertir de nuevo a formato de fecha
        self.df["Fecha Hora"] = pd.to_datetime(self.df["Fecha Hora"], errors='coerce')
        
        # Filtrar por fecha (últimos 45 días)
        fecha_limite = datetime.now() - timedelta(days=45)
        self.df = self.df[self.df['Fecha Hora'] >= fecha_limite]
        
        # Procesar nombres de equipos
        self.df['Equipo'] = self.df['Equipo'].str.replace('_CDV', 'CDV')
        
        # Separar columna Equipo
        split_df = self.df['Equipo'].str.split('_', expand=True)
        split_df = split_df.drop(columns=[0])
        split_df.columns = ['CDV', 'Estacion']
        self.df = pd.concat([self.df, split_df], axis=1)
        self.df = self.df.drop(columns=['Equipo'])
        self.df['Equipo'] = self.df['Estacion'] + '_CDV_' + self.df['CDV']
        self.df = self.df.drop("CDV", axis=1)
        
        if progress_callback:
            progress_callback(40, "Preprocesamiento completado")
        
        return True
    
    def calculate_time_differences(self, progress_callback=None):
        """Calcular diferencias temporales entre eventos"""
        if progress_callback:
            progress_callback(40, "Calculando diferencias temporales...")
        
        # Diferencias con registros anteriores
        self.df["Diff.Time_-1_row"] = self.df["Fecha Hora"].diff(periods=1)
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].dt.total_seconds()
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].astype("float64")
        self.df["Diff.Time_-1_row"] = round(self.df["Diff.Time_-1_row"], 1)
        
        self.df["Diff.Time_-2_row"] = self.df["Fecha Hora"].diff(periods=2)
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].dt.total_seconds()
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].astype("float64")
        self.df["Diff.Time_-2_row"] = round(self.df["Diff.Time_-2_row"], 1)
        
        # Diferencias con registros siguientes
        self.df["Diff.Time_+1_row"] = self.df["Fecha Hora"].diff(periods=-1)
        self.df["Diff.Time_+1_row"] = -1 * self.df["Diff.Time_+1_row"].dt.total_seconds()
        self.df["Diff.Time_+1_row"] = self.df["Diff.Time_+1_row"].astype("float64")
        self.df["Diff.Time_+1_row"] = round(self.df["Diff.Time_+1_row"], 1)
        
        self.df["Diff.Time_+2_row"] = self.df["Fecha Hora"].diff(periods=-2)
        self.df["Diff.Time_+2_row"] = -1 * self.df["Diff.Time_+2_row"].dt.total_seconds()
        self.df["Diff.Time_+2_row"] = self.df["Diff.Time_+2_row"].astype("float64")
        self.df["Diff.Time_+2_row"] = round(self.df["Diff.Time_+2_row"], 1)
        
        # Filtrar tiempos válidos
        self.df = self.df.loc[self.df["Diff.Time_-1_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_-2_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_+1_row"] >= 0.0]
        self.df = self.df.loc[self.df["Diff.Time_+2_row"] >= 0.0]
        
        # Calcular tiempo conjunto
        self.df["Tiempo Conjunto"] = self.df["Diff.Time_-1_row"] + (self.df["Diff.Time_+2_row"])
        self.df["Tiempo Conjunto"] = self.df["Tiempo Conjunto"].astype("float64")
        self.df["Tiempo Conjunto"] = round(self.df["Tiempo Conjunto"], 2)
        
        if progress_callback:
            progress_callback(60, "Cálculo de diferencias temporales completado")
        
        return True
    
    def calculate_statistics(self, progress_callback=None):
        """Calcular estadísticas para cada equipo y estado"""
        if progress_callback:
            progress_callback(60, "Calculando estadísticas...")
        
        # Copias para análisis
        df_L1_lb = self.df.copy()
        df_L1_oc = self.df.copy()
        
        # Estadísticas generales
        df_L1_aux_avg = self.df.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_+1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        
        # Estadísticas para estado 0 (liberación)
        df_L1_lb["Estado"] = df_L1_lb["Estado"].astype(str)
        df_L1_lb = df_L1_lb.loc[df_L1_lb["Estado"] == "0"]
        df_L1_aux_lb = df_L1_lb.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        df_L1_aux_lb.rename(
            columns={"mean": "mean_lib", "std": "std_lib", "median": "median_lib"},
            inplace=True
        )
        
        # Estadísticas para estado 1 (ocupación)
        df_L1_oc["Estado"] = df_L1_oc["Estado"].astype(str)
        df_L1_oc = df_L1_oc.loc[df_L1_oc["Estado"] == "1"]
        df_L1_aux_oc = df_L1_oc.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        df_L1_aux_oc.rename(
            columns={"mean": "mean_oc", "std": "std_oc", "median": "median_oc"},
            inplace=True
        )
        
        # Combinar estadísticas
        self.df = self.df.merge(df_L1_aux_lb, on="Equipo")
        self.df = self.df.merge(df_L1_aux_oc, on="Equipo")
        
        # Ordenar
        self.df = self.df.sort_values(["Equipo", "Fecha Hora"])
        
        # Limpiar ID y redondear
        self.df = self.df.drop("ID", axis=1)
        self.df['mean_lib'] = self.df['mean_lib'].round(1)
        self.df['std_lib'] = self.df['std_lib'].round(1)
        self.df['std_oc'] = self.df['std_oc'].round(1)
        self.df['mean_oc'] = self.df['mean_oc'].round(1)
        
        # Filtrar por horario operativo
        self.df.set_index('Fecha Hora', inplace=True)
        self.df = self.df[(self.df.index.hour >= 6) & (self.df.index.hour <= 23)]
        self.df = self.df.reset_index()
        
        if progress_callback:
            progress_callback(70, "Cálculo de estadísticas completado")
        
        return True
    
    def detect_anomalies(self, progress_callback=None):
        """Detectar anomalías basadas en umbrales estadísticos"""
        if progress_callback:
            progress_callback(70, "Detectando anomalías...")
        
        # Asegurar valores absolutos para las diferencias
        self.df['Diff.Time_+1_row'] = self.df['Diff.Time_+1_row'].abs()
        
        # Preparar condiciones
        self.df["Estado"] = self.df["Estado"].astype(str)
        estado_0 = self.df["Estado"] == "0"
        estado_1 = self.df["Estado"] == "1"
        
        # Detectar Fallos de Ocupación (FO)
        self.df["FO"] = np.where(
            estado_0, 
            np.where(self.df["Diff.Time_+1_row"] < (self.f_oc_1 * self.df["median_oc"]), "PFO", "NFO"), 
            "NA"
        )
        
        # Detectar Fallos de Liberación (FL)
        self.df["FL"] = np.where(
            estado_1, 
            np.where(self.df["Diff.Time_+1_row"] < (self.f_lb_2 * self.df["median_lib"]), "PFL", "NFL"), 
            "NA"
        )
        
        # Agregar identificador de línea
        self.df["Linea"] = "L1"
        
        if progress_callback:
            progress_callback(80, "Detección de anomalías completada")
        
        return True
    
    def prepare_reports(self, progress_callback=None):
        """Preparar los diferentes reportes"""
        if progress_callback:
            progress_callback(80, "Preparando reportes...")
        
        # 1. Preparar reporte de fallos de ocupación (FO)
        self.df_L1_FO = self.df.loc[self.df['FO'] == 'PFO']
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
                           'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
                           'mean_oc', 'median_oc', 'std_oc', 'FL']
        self.df_L1_FO = self.df_L1_FO.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L1_FO['Fecha Hora'] = self.df_L1_FO['Fecha Hora'].astype(str)
        self.df_L1_FO['Equipo'] = self.df_L1_FO['Equipo'].astype(str)
        self.df_L1_FO['Diff.Time_+1_row'] = self.df_L1_FO['Diff.Time_+1_row'].astype(str)
        self.df_L1_FO['ID'] = self.df_L1_FO['Fecha Hora'] + self.df_L1_FO['Equipo'] + self.df_L1_FO['Diff.Time_+1_row']
        
        # 2. Preparar reporte de ocupaciones por día
        df = self.df.copy()
        df['Fecha'] = df['Fecha Hora'].dt.to_period('D')
        df_pfo = df[df['FO'] == 'PFO']
        pfo_count = df_pfo.groupby(['Fecha', 'Equipo']).size().reset_index(name='PFO')
        pfo_na_count = df.groupby(['Fecha', 'Equipo']).size().reset_index(name='OCUPACIONES')
        
        pfo_count['PFO'] = pfo_count['PFO'].astype(int)
        pfo_na_count['OCUPACIONES'] = pfo_na_count['OCUPACIONES'].astype(int)
        
        self.df_L1_OCUP = pd.concat([pfo_count, pfo_na_count], ignore_index=True)
        self.df_L1_OCUP['PFO'].fillna(0, inplace=True)
        self.df_L1_OCUP['OCUPACIONES'].fillna(0, inplace=True)
        self.df_L1_OCUP['Count'] = self.df_L1_OCUP['OCUPACIONES'] / 2
        self.df_L1_OCUP.drop(columns=['OCUPACIONES'], inplace=True)
        self.df_L1_OCUP['Count'] = self.df_L1_OCUP['Count'].astype(int)
        self.df_L1_OCUP["Linea"] = "L1"
        
        # Eliminar PFO si existe y crear ID único
        self.df_L1_OCUP = self.df_L1_OCUP.drop(columns=['PFO'])
        self.df_L1_OCUP['Fecha'] = self.df_L1_OCUP['Fecha'].astype(str)
        self.df_L1_OCUP['Count'] = self.df_L1_OCUP['Count'].astype(str)
        self.df_L1_OCUP['Equipo'] = self.df_L1_OCUP['Equipo'].astype(str)
        self.df_L1_OCUP['ID'] = self.df_L1_OCUP['Equipo'] + "*" + self.df_L1_OCUP['Fecha'] + "*" + self.df_L1_OCUP['Count'] + "*" + self.df_L1_OCUP['Linea']
        
        # 3. Preparar reporte de fallos de liberación (FL)
        self.df_L1_FL = self.df.loc[self.df['FL'] == 'PFL']
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
                          'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
                          'mean_oc', 'median_oc', 'std_oc', 'FO', 'Fecha']
        self.df_L1_FL = self.df_L1_FL.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L1_FL['Fecha Hora'] = self.df_L1_FL['Fecha Hora'].astype(str)
        self.df_L1_FL['Equipo'] = self.df_L1_FL['Equipo'].astype(str)
        self.df_L1_FL['Diff.Time_+1_row'] = self.df_L1_FL['Diff.Time_+1_row'].astype(str)
        self.df_L1_FL['ID'] = self.df_L1_FL['Fecha Hora'] + self.df_L1_FL['Equipo'] + self.df_L1_FL['Diff.Time_+1_row']
        
        if progress_callback:
            progress_callback(85, "Preparación de reportes completada")
        
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(85, "Iniciando actualización de reportes...")
            
            # 1. Actualizar reporte de fallos de ocupación
            fo_file_path = os.path.join(self.output_folder_path, 'df_L1_FO_Mensual.csv')
            if os.path.exists(fo_file_path):
                df_L1_FO_Mensual = pd.read_csv(fo_file_path)
                
                # Concatenar y eliminar duplicados
                df_L1_FO_Mensual = pd.concat([df_L1_FO_Mensual, self.df_L1_FO], ignore_index=True)
                df_L1_FO_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L1_FO_Mensual.to_csv(fo_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L1_FO.to_csv(fo_file_path, index=False)
            
            # 2. Actualizar reporte de ocupaciones
            ocup_file_path = os.path.join(self.output_folder_path, 'df_L1_OCUP_Mensual.csv')
            if os.path.exists(ocup_file_path):
                df_L1_OCUP_Mensual = pd.read_csv(ocup_file_path)
                
                # Concatenar y eliminar duplicados
                df_L1_OCUP_Mensual = pd.concat([df_L1_OCUP_Mensual, self.df_L1_OCUP], ignore_index=True)
                df_L1_OCUP_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L1_OCUP_Mensual.to_csv(ocup_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L1_OCUP.to_csv(ocup_file_path, index=False)
            
            # 3. Actualizar reporte de fallos de liberación
            fl_file_path = os.path.join(self.output_folder_path, 'df_L1_FL_Mensual.csv')
            if os.path.exists(fl_file_path):
                df_L1_FL_Mensual = pd.read_csv(fl_file_path)
                
                # Concatenar y eliminar duplicados
                df_L1_FL_Mensual = pd.concat([df_L1_FL_Mensual, self.df_L1_FL], ignore_index=True)
                df_L1_FL_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L1_FL_Mensual.to_csv(fl_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L1_FL.to_csv(fl_file_path, index=False)
            
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
            fecha_limite = datetime.now() - timedelta(days=45)
            self.df = self.df[self.df['Fecha Hora'] >= fecha_limite]
            
            main_file_path = os.path.join(self.output_folder_path, 'df_L1_CDV.csv')
            self.df.to_csv(main_file_path, index=True)
            return True
        except Exception as e:
            return False