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
                    df = pd.read_excel(file_path)
                
                # Verificar si es un archivo con formato esperado
                if 'FECHA' in df.columns and 'HORA' in df.columns:
                    # Filtrar columnas que empiezan con 'CDV'
                    cdv_cols = [col for col in df.columns if col.startswith('CDV')]
                    if not cdv_cols:
                        continue
                    
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
                    df_melted['Fecha Hora'] = pd.to_datetime(
                        df_melted['FECHA'].astype(str) + ' ' + df_melted['HORA'].astype(str),
                        errors='coerce'
                    )
                    
                    # Modificar Estado: considerar 1 como "Ocupacion" y 0 como "Liberacion"
                    df_melted['Estado'] = df_melted['Estado'].apply(
                        lambda x: 'Ocupacion' if x == 0 else 'Liberacion' if x == 1 else 'Desconocido'
                    )
                    
                    # Seleccionar columnas finales
                    df_melted = df_melted[['Fecha Hora', 'Equipo', 'Estacion', 'Subsistema', 'Estado']]
                    
                    # Agregar al listado
                    df_list.append(df_melted)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(None, f"Error en archivo {file_path}: {str(e)}")
        
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
        
        # Convertir estados a numéricos
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
            self.df.to_csv(main_file_path, index=False)
            return True
        except Exception as e:
            return False
    
    def process_data(self, progress_callback=None):
        """Ejecutar todo el proceso de análisis de datos"""
        try:
            # 1. Encontrar archivos CSV/Excel
            if progress_callback:
                progress_callback(0, "Buscando archivos CSV/Excel...")
            num_files = self.find_files()
            if num_files == 0:
                if progress_callback:
                    progress_callback(100, "No se encontraron archivos CSV/Excel para procesar")
                return False
            
            # 2. Leer archivos CSV/Excel
            if progress_callback:
                progress_callback(5, f"Leyendo {num_files} archivos...")
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
                progress_callback(100, "Procesamiento CDV Línea 2 completado con éxito")
            
            return True
        
        except Exception as e:
            if progress_callback:
                progress_callback(None, f"Error en el procesamiento: {str(e)}")
            return False