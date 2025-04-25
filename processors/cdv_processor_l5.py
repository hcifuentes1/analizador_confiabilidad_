# processors/cdv_processor_l5.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from processors.base_processor import BaseProcessor

class CDVProcessorL5(BaseProcessor):
    """Procesador para datos CDV de la Línea 5"""
    
    def __init__(self):
        super().__init__(line="L5", analysis_type="CDV")
        # Atributos específicos para CDV L5
        self.df_L5_2 = None
        self.df_L5_FO = None
        self.df_L5_FL = None
        self.df_L5_OCUP = None
        
        # Factores de umbral para detección de anomalías
        self.f_oc_1 = 0.1
        self.f_lb_2 = 0.05
    
    def find_files(self):
        """Encontrar archivos TXT para análisis CDV de Línea 5"""
        self.txt_files = []
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
        """Leer archivos TXT para análisis CDV de Línea 5"""
        df_list = []
        total_files = len(self.txt_files)
        
        for i, txt in enumerate(self.txt_files):
            try:
                if progress_callback:
                    progress = (i / total_files) * 15
                    progress_callback(5 + progress, f"Procesando archivo {i+1} de {total_files}: {os.path.basename(txt)}")
                
                df = pd.read_csv(txt, encoding="Latin-1", sep=";", skiprows=7, header=None, engine='python')
                df = df[df[4].str.contains('CDV', na=False)]
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
        
        if progress_callback:
            progress_callback(25, "Formateando datos...")
            
        # Eliminar columnas innecesarias y reordenar
        self.df = self.df.drop(self.df.columns[[0,1,6,7]], axis=1)
        cols = [4,2,0,1,3]
        self.df = self.df.iloc[:,cols]
        
        # Renombrar columnas
        self.df.rename(
            columns={"Fecha Hora": "Fecha Hora", 4: "Equipo", 2: "Estacion", 3: "Subsistema", 5: "Estado"},
            inplace=True,
        )
        
        if progress_callback:
            progress_callback(30, "Filtrando por fecha...")
            
        # Filtrar por fecha (últimos 40 días)
        date_threshold = datetime.now() - timedelta(days=40)
        self.df = self.df[self.df["Fecha Hora"] >= date_threshold]
        
        if progress_callback:
            progress_callback(35, "Ordenando datos...")
            
        # Ordenar y filtrar por hora del día
        self.df = self.df.sort_values(["Equipo", "Fecha Hora"])
        self.df = self.df.reset_index()
        self.df["Estado"] = self.df["Estado"].str.split(" ").str[0]
        self.df.set_index('Fecha Hora', inplace=True)
        self.df = self.df[(self.df.index.hour >= 6) & (self.df.index.hour <= 23)]
        self.df = self.df.reset_index()
        
        if progress_callback:
            progress_callback(40, "Procesando estados...")
            
        # Procesar estados
        self.process_states()
        
        if progress_callback:
            progress_callback(45, "Preprocesamiento completado")
        
        return True
    
    def process_states(self):
        """Procesar estados para CDV de Línea 5"""
        # Crear copia del DataFrame
        self.df_L5_2 = self.df.copy()
        
        # Convertir estados a numéricos
        self.df_L5_2['Estado'] = self.df_L5_2['Estado'].replace('Liberacion', 1)
        self.df_L5_2['Estado'] = self.df_L5_2['Estado'].replace('Ocupacion', 0)
        
        # Ordenar y filtrar
        self.df_L5_2 = self.df_L5_2.sort_values(["Equipo", "Fecha Hora"])
        self.df_L5_2 = self.df_L5_2[self.df_L5_2['Estado'].isin([1, 0])]
        self.df_L5_2["Estado"] = self.df_L5_2["Estado"].astype("float64")
        
        # Detectar cambios de estado
        self.df_L5_2["Diff_Aux"] = self.df_L5_2["Estado"].diff(periods=1)
        self.df_L5_2["Diff_Aux"] = self.df_L5_2["Diff_Aux"].astype("string")
        self.df_L5_2 = self.df_L5_2.loc[~(self.df_L5_2["Diff_Aux"].str.contains("0.0"))]
        
        # Limpiar y restaurar etiquetas originales
        self.df_L5_2 = self.df_L5_2.drop('Diff_Aux', axis=1)
        self.df_L5_2['Estado'] = self.df_L5_2['Estado'].replace(1, 'Liberacion')
        self.df_L5_2['Estado'] = self.df_L5_2['Estado'].replace(0, 'Ocupacion')
        
        # Actualizar DataFrame principal
        self.df = self.df_L5_2.copy()
    
    def calculate_time_differences(self, progress_callback=None):
        """Calcular diferencias de tiempo entre registros"""
        if progress_callback:
            progress_callback(45, "Calculando diferencias temporales...")
            
        # Calcular diferencia con registros anteriores
        self.df["Diff.Time_-1_row"] = self.df["Fecha Hora"].diff(periods=1)
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].dt.total_seconds()
        self.df["Diff.Time_-1_row"] = self.df["Diff.Time_-1_row"].astype("float64")
        self.df["Diff.Time_-1_row"] = round(self.df["Diff.Time_-1_row"], 1)
        
        self.df["Diff.Time_-2_row"] = self.df["Fecha Hora"].diff(periods=2)
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].dt.total_seconds()
        self.df["Diff.Time_-2_row"] = self.df["Diff.Time_-2_row"].astype("float64")
        self.df["Diff.Time_-2_row"] = round(self.df["Diff.Time_-2_row"], 1)
        
        if progress_callback:
            progress_callback(50, "Calculando diferencias temporales hacia adelante...")
            
        # Calcular diferencia con registros siguientes
        self.df["Diff.Time_+1_row"] = self.df["Fecha Hora"].diff(periods=-1)
        self.df["Diff.Time_+1_row"] = -1 * self.df["Diff.Time_+1_row"].dt.total_seconds()
        self.df["Diff.Time_+1_row"] = self.df["Diff.Time_+1_row"].astype("float64")
        self.df["Diff.Time_+1_row"] = round(self.df["Diff.Time_+1_row"], 1)
        
        self.df["Diff.Time_+2_row"] = self.df["Fecha Hora"].diff(periods=-2)
        self.df["Diff.Time_+2_row"] = -1 * self.df["Diff.Time_+2_row"].dt.total_seconds()
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
        df_L5_lb = self.df.copy()
        df_L5_oc = self.df.copy()
        
        if progress_callback:
            progress_callback(67, "Calculando estadísticas generales...")
            
        # Calcular estadísticas generales
        df_L5_aux_avg = self.df.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_+1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        
        if progress_callback:
            progress_callback(69, "Calculando estadísticas para liberación...")
            
        # Estadísticas para liberación
        df_L5_lb = df_L5_lb.loc[df_L5_lb["Estado"].str.contains("Ocupacion")]
        df_L5_aux_lb = df_L5_lb.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        df_L5_aux_lb.rename(
            columns={"mean": "mean_lib", "std": "std_lib", "median": "median_lib"},
            inplace=True
        )
        
        if progress_callback:
            progress_callback(71, "Calculando estadísticas para ocupación...")
            
        # Estadísticas para ocupación
        df_L5_oc = df_L5_oc.loc[df_L5_oc["Estado"].str.contains("Liberacion")]
        df_L5_aux_oc = df_L5_oc.pivot_table(
            index=["Equipo", "Estado"],
            values="Diff.Time_-1_row",
            aggfunc={np.mean, np.std, np.median}
        )
        df_L5_aux_oc.rename(
            columns={"mean": "mean_oc", "std": "std_oc", "median": "median_oc"},
            inplace=True
        )
        
        if progress_callback:
            progress_callback(73, "Combinando estadísticas...")
            
        # Combinar estadísticas con el DataFrame principal
        self.df = self.df.merge(df_L5_aux_lb, on="Equipo")
        self.df = self.df.merge(df_L5_aux_oc, on="Equipo")
        
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
        self.df["Linea"] = "L5"
        
        if progress_callback:
            progress_callback(85, "Detección de anomalías completada")
        
        return True
    
    def prepare_reports(self, progress_callback=None):
        """Preparar los diferentes reportes"""
        if progress_callback:
            progress_callback(85, "Iniciando preparación de reportes...")
            
        # 1. Preparar reporte de fallos de ocupación (FO)
        self.df_L5_FO = self.df.loc[self.df['FO'] == 'PFO']
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
                           'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
                           'mean_oc', 'median_oc', 'std_oc', 'FL']
        self.df_L5_FO = self.df_L5_FO.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L5_FO['Fecha Hora'] = self.df_L5_FO['Fecha Hora'].astype(str)
        self.df_L5_FO['Equipo'] = self.df_L5_FO['Equipo'].astype(str)
        self.df_L5_FO['Diff.Time_+1_row'] = self.df_L5_FO['Diff.Time_+1_row'].astype(str)
        self.df_L5_FO['ID'] = self.df_L5_FO['Fecha Hora'] + self.df_L5_FO['Equipo'] + self.df_L5_FO['Diff.Time_+1_row']
        
        if progress_callback:
            progress_callback(87, "Preparando reporte de ocupaciones...")
            
        # 2. Preparar reporte de conteo de ocupaciones
        # Extraer fecha (sin hora)
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha Hora']).dt.date
        
        # Filtrar y agrupar
        self.df_L5_OCUP = self.df[self.df['Estado'] == 'Ocupacion']
        self.df_L5_OCUP = self.df_L5_OCUP.groupby(['Equipo', 'Fecha']).size().reset_index(name='Count')
        
        # Crear ID único
        self.df_L5_OCUP['Fecha'] = self.df_L5_OCUP['Fecha'].astype(str)
        self.df_L5_OCUP['Count'] = self.df_L5_OCUP['Count'].astype(str)
        self.df_L5_OCUP['Equipo'] = self.df_L5_OCUP['Equipo'].astype(str)
        self.df_L5_OCUP['ID'] = self.df_L5_OCUP['Fecha'] + self.df_L5_OCUP['Equipo'] + self.df_L5_OCUP['Count']
        
        if progress_callback:
            progress_callback(89, "Preparando reporte de fallos de liberación...")
            
        # 3. Preparar reporte de fallos de liberación (FL)
        self.df_L5_FL = self.df.loc[self.df['FL'] == 'PFL']
        
        # Eliminar columnas innecesarias
        columns_to_drop = ['Estado', 'Diff.Time_-1_row', 'Diff.Time_-2_row', 'Diff.Time_+2_row', 
                          'Tiempo Conjunto', 'mean_lib', 'median_lib', 'std_lib', 
                          'mean_oc', 'median_oc', 'std_oc', 'FO', 'Fecha']
        self.df_L5_FL = self.df_L5_FL.drop(columns=columns_to_drop)
        
        # Crear ID único
        self.df_L5_FL['Fecha Hora'] = self.df_L5_FL['Fecha Hora'].astype(str)
        self.df_L5_FL['Equipo'] = self.df_L5_FL['Equipo'].astype(str)
        self.df_L5_FL['Diff.Time_+1_row'] = self.df_L5_FL['Diff.Time_+1_row'].astype(str)
        self.df_L5_FL['ID'] = self.df_L5_FL['Fecha Hora'] + self.df_L5_FL['Equipo'] + self.df_L5_FL['Diff.Time_+1_row']
        
        if progress_callback:
            progress_callback(90, "Preparación de reportes completada")
        
        return True
    
    def update_reports(self, progress_callback=None):
        """Actualizar los reportes existentes con nuevos datos"""
        try:
            if progress_callback:
                progress_callback(90, "Iniciando actualización de reportes...")
                
            # 1. Actualizar reporte de fallos de ocupación
            fo_file_path = os.path.join(self.output_folder_path, 'df_L5_FO_Mensual.csv')
            if os.path.exists(fo_file_path):
                df_L5_FO_Mensual = pd.read_csv(fo_file_path)
                
                # Concatenar y eliminar duplicados
                df_L5_FO_Mensual = pd.concat([df_L5_FO_Mensual, self.df_L5_FO], ignore_index=True)
                df_L5_FO_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L5_FO_Mensual.to_csv(fo_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L5_FO.to_csv(fo_file_path, index=False)
            
            if progress_callback:
                progress_callback(93, "Actualizando reporte de ocupaciones...")
                
            # 2. Actualizar reporte de conteo de ocupaciones
            ocup_file_path = os.path.join(self.output_folder_path, 'df_L5_OCUP_Mensual.csv')
            if os.path.exists(ocup_file_path):
                df_L5_OCUP_Mensual = pd.read_csv(ocup_file_path)
                
                # Concatenar y eliminar duplicados
                df_L5_OCUP_Mensual = pd.concat([df_L5_OCUP_Mensual, self.df_L5_OCUP], ignore_index=True)
                df_L5_OCUP_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L5_OCUP_Mensual.to_csv(ocup_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L5_OCUP.to_csv(ocup_file_path, index=False)
            
            if progress_callback:
                progress_callback(96, "Actualizando reporte de fallos de liberación...")
                
            # 3. Actualizar reporte de fallos de liberación
            fl_file_path = os.path.join(self.output_folder_path, 'df_L5_FL_Mensual.csv')
            if os.path.exists(fl_file_path):
                df_L5_FL_Mensual = pd.read_csv(fl_file_path)
                
                # Concatenar y eliminar duplicados
                df_L5_FL_Mensual = pd.concat([df_L5_FL_Mensual, self.df_L5_FL], ignore_index=True)
                df_L5_FL_Mensual.drop_duplicates(subset=['ID'], inplace=True)
                
                # Guardar el resultado actualizado
                df_L5_FL_Mensual.to_csv(fl_file_path, index=False)
            else:
                # Si el archivo no existe, guardar el nuevo
                self.df_L5_FL.to_csv(fl_file_path, index=False)
            
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
            main_file_path = os.path.join(self.output_folder_path, 'df_L5_CDV.csv')
            self.df.to_csv(main_file_path, index=True)
            return True
        except Exception as e:
            return False
    
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos TXT
            if progress_callback:
                progress_callback(0, "Buscando archivos TXT...")
            num_files = self.find_files()
            if num_files == 0:
                if progress_callback:
                    progress_callback(100, "No se encontraron archivos TXT para procesar")
                return False
            
            # 2. Leer archivos TXT
            if progress_callback:
                progress_callback(5, f"Leyendo {num_files} archivos TXT...")
            if not self.read_files(progress_callback):
                return False
            
            # 3. Preprocesar datos
            if progress_callback:
                progress_callback(20, "Preprocesando datos...")
            self.preprocess_data(progress_callback)
            
            # 4. Calcular diferencias temporales
            if progress_callback:
                progress_callback(45, "Calculando diferencias temporales...")
            self.calculate_time_differences(progress_callback)
            
            # 5. Calcular estadísticas
            if progress_callback:
                progress_callback(65, "Calculando estadísticas...")
            self.calculate_statistics(progress_callback)
            
            # 6. Detectar anomalías
            if progress_callback:
                progress_callback(75, "Detectando anomalías...")
            self.detect_anomalies(progress_callback)
            
            # 7. Preparar reportes
            if progress_callback:
                progress_callback(85, "Preparando reportes...")
            self.prepare_reports(progress_callback)
            
            # 8. Actualizar reportes existentes
            if progress_callback:
                progress_callback(90, "Actualizando reportes existentes...")
            self.update_reports(progress_callback)
            
            # 9. Guardar DataFrame principal
            if progress_callback:
                progress_callback(98, "Guardando DataFrame principal...")
            self.save_dataframe()
            
            if progress_callback:
                progress_callback(100, "Procesamiento CDV Línea 5 completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False