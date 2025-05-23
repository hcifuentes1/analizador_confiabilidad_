# dashboard/dashboard_generator.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import webbrowser
import threading
import logging
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardGenerator:
    """Generador de dashboard web interactivo para visualizar resultados del análisis"""
    
    def __init__(self, output_folder, line, analysis_type, port=8050):
        self.output_folder = output_folder
        self.line = line
        self.analysis_type = analysis_type
        
        # Intentar encontrar un puerto disponible si el especificado está en uso
        import socket
        self.port = port
        max_port_attempts = 10
        for attempt in range(max_port_attempts):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('127.0.0.1', self.port))
                s.close()
                break  # Si llegamos aquí, el puerto está disponible
            except:
                logger.warning(f"Puerto {self.port} no disponible, intentando con el siguiente puerto...")
                self.port += 1
                if attempt == max_port_attempts - 1:
                    logger.warning(f"No se pudo encontrar un puerto disponible después de {max_port_attempts} intentos")
        
        self.dataframes = {}
        self.app = None
        self.server_thread = None
        self.running = False
        self.insights = None  # Para almacenar los insights entre actualizaciones
        self.reliability_metrics = {}  # Para almacenar métricas de confiabilidad
        
        # Colores para las gráficas
        self.colors = {
            'background': '#F0F2F6',
            'card_background': '#FFFFFF',
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#2ECC71',
            'info': '#3498DB',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'text': '#2C3E50'
        }
        
        self.line_colors = {
            'L1': '#FF0000',  # Rojo
            'L2': '#FFCC00',  # Amarillo
            'L4': '#0066CC',  # Azul
            'L4A': '#9900CC',  # Morado
            'L5': '#009933'   # Verde
        }
    
    def load_data(self):
        """Cargar datos desde los archivos CSV generados"""
        try:
            if self.analysis_type == "CDV":
                # Cargar archivos CDV
                fo_file_path = os.path.join(self.output_folder, f'df_{self.line}_FO_Mensual.csv')
                fl_file_path = os.path.join(self.output_folder, f'df_{self.line}_FL_Mensual.csv')
                ocup_file_path = os.path.join(self.output_folder, f'df_{self.line}_OCUP_Mensual.csv')
                main_file_path = os.path.join(self.output_folder, f'df_{self.line}_CDV.csv')
                
                if os.path.exists(fo_file_path):
                    self.dataframes['fallos_ocupacion'] = pd.read_csv(fo_file_path)
                    # Convertir fechas
                    if 'Fecha Hora' in self.dataframes['fallos_ocupacion'].columns:
                        self.dataframes['fallos_ocupacion']['Fecha Hora'] = pd.to_datetime(
                            self.dataframes['fallos_ocupacion']['Fecha Hora'], errors='coerce')
                
                if os.path.exists(fl_file_path):
                    self.dataframes['fallos_liberacion'] = pd.read_csv(fl_file_path)
                    # Convertir fechas
                    if 'Fecha Hora' in self.dataframes['fallos_liberacion'].columns:
                        self.dataframes['fallos_liberacion']['Fecha Hora'] = pd.to_datetime(
                            self.dataframes['fallos_liberacion']['Fecha Hora'], errors='coerce')
                
                if os.path.exists(ocup_file_path):
                    self.dataframes['ocupaciones'] = pd.read_csv(ocup_file_path)
                    # Convertir fechas
                    if 'Fecha' in self.dataframes['ocupaciones'].columns:
                        self.dataframes['ocupaciones']['Fecha'] = pd.to_datetime(
                            self.dataframes['ocupaciones']['Fecha'], errors='coerce')
                
                if os.path.exists(main_file_path):
                    self.dataframes['main'] = pd.read_csv(main_file_path)
                    # Convertir fechas
                    if 'Fecha Hora' in self.dataframes['main'].columns:
                        self.dataframes['main']['Fecha Hora'] = pd.to_datetime(
                            self.dataframes['main']['Fecha Hora'], errors='coerce')
                
            elif self.analysis_type == "ADV":
                # Cargar archivos ADV
                disc_file_path = os.path.join(self.output_folder, f'df_{self.line}_ADV_DISC_Mensual.csv')
                mov_file_path = os.path.join(self.output_folder, f'df_{self.line}_ADV_MOV_Mensual.csv')
                
                if os.path.exists(disc_file_path):
                    self.dataframes['discordancias'] = pd.read_csv(disc_file_path)
                    # Convertir fechas
                    if 'Fecha Hora' in self.dataframes['discordancias'].columns:
                        self.dataframes['discordancias']['Fecha Hora'] = pd.to_datetime(
                            self.dataframes['discordancias']['Fecha Hora'], dayfirst=True, errors='coerce')
                
                if os.path.exists(mov_file_path):
                    self.dataframes['movimientos'] = pd.read_csv(mov_file_path)
                    # Convertir fechas
                    if 'Fecha' in self.dataframes['movimientos'].columns:
                        self.dataframes['movimientos']['Fecha'] = pd.to_datetime(
                            self.dataframes['movimientos']['Fecha'], errors='coerce')
            
            # Verificar si se cargaron datos
            if any(not df.empty for df in self.dataframes.values()):
                # Generar los insights iniciales
                self.insights = self.generate_insights()
                logger.info(f"Insights generados para {self.line} - {self.analysis_type}")
            
            logger.info(f"Datos cargados exitosamente para {self.line} - {self.analysis_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar los datos: {str(e)}")
            return False
    
    def generate_insights(self, filtered_dataframes=None):
        """Generar insights y recomendaciones basadas en el análisis de datos"""
        # Usar dataframes filtrados si se proporcionan
        dfs = filtered_dataframes if filtered_dataframes else self.dataframes
        
        insights = {
            'recomendaciones_preventivas': [],
            'recomendaciones_predictivas': [],
            'alertas_urgentes': [], 
            'patrones_detectados': [],
            'anomalias': [],
            'resumen': {},
            'metricas_confiabilidad': {}
        }
        
        try:
            if self.analysis_type == "CDV":
                # Calcular métricas de confiabilidad para cada equipo
                if 'fallos_ocupacion' in dfs and not dfs['fallos_ocupacion'].empty:
                    reliability_fo = self.calculate_equipment_reliability(
                        dfs['fallos_ocupacion'], 
                        equipment_col='Equipo', 
                        failure_date_col='Fecha Hora'
                    )
                    
                    # Guardar en insights
                    insights['metricas_confiabilidad']['fallos_ocupacion'] = reliability_fo
                
                if 'fallos_liberacion' in dfs and not dfs['fallos_liberacion'].empty:
                    reliability_fl = self.calculate_equipment_reliability(
                        dfs['fallos_liberacion'], 
                        equipment_col='Equipo', 
                        failure_date_col='Fecha Hora'
                    )
                    
                    # Guardar en insights
                    insights['metricas_confiabilidad']['fallos_liberacion'] = reliability_fl
                
                # Combinar y analizar métricas de confiabilidad
                all_equipment = set()
                combined_metrics = {}
                
                if 'fallos_ocupacion' in insights['metricas_confiabilidad']:
                    all_equipment.update(insights['metricas_confiabilidad']['fallos_ocupacion'].keys())
                    combined_metrics.update(insights['metricas_confiabilidad']['fallos_ocupacion'])
                
                if 'fallos_liberacion' in insights['metricas_confiabilidad']:
                    all_equipment.update(insights['metricas_confiabilidad']['fallos_liberacion'].keys())
                    
                    # Combinar métricas de liberación con ocupación
                    for equip, metrics in insights['metricas_confiabilidad']['fallos_liberacion'].items():
                        if equip in combined_metrics:
                            combined_metrics[equip]['total_failures'] += metrics['total_failures']
                            combined_metrics[equip]['recent_failures'] += metrics['recent_failures']
                            combined_metrics[equip]['is_increasing'] = combined_metrics[equip]['is_increasing'] or metrics['is_increasing']
                            
                            # Usar la fecha más reciente de fallo
                            if metrics['last_failure_date'] > combined_metrics[equip]['last_failure_date']:
                                combined_metrics[equip]['last_failure_date'] = metrics['last_failure_date']
                                combined_metrics[equip]['days_since_last_failure'] = metrics['days_since_last_failure']
                            
                            # Aumentar urgencia
                            combined_metrics[equip]['maintenance_urgency'] = min(
                                1.0, 
                                combined_metrics[equip]['maintenance_urgency'] + metrics['maintenance_urgency']
                            )
                        else:
                            combined_metrics[equip] = metrics
                
                # Generar recomendaciones predictivas
                high_urgency_equips = []
                medium_urgency_equips = []
                low_urgency_equips = []
                
                for equip, metrics in combined_metrics.items():
                    urgency = metrics['maintenance_urgency']
                    
                    # Predecir tendencia para equipos de alta frecuencia
                    if metrics['total_failures'] >= 5:
                        trend_fo = None
                        trend_fl = None
                        
                        if 'fallos_ocupacion' in dfs and equip in insights['metricas_confiabilidad'].get('fallos_ocupacion', {}):
                            trend_fo = self.predict_failure_trend(
                                dfs['fallos_ocupacion'],
                                equipment_col='Equipo',
                                date_col='Fecha Hora',
                                equipment=equip
                            )
                        
                        if 'fallos_liberacion' in dfs and equip in insights['metricas_confiabilidad'].get('fallos_liberacion', {}):
                            trend_fl = self.predict_failure_trend(
                                dfs['fallos_liberacion'],
                                equipment_col='Equipo',
                                date_col='Fecha Hora',
                                equipment=equip
                            )
                        
                        # Agregar predicción a las métricas
                        if trend_fo or trend_fl:
                            combined_metrics[equip]['prediccion'] = trend_fo if trend_fo else trend_fl
                    
                    # Clasificar por urgencia
                    if urgency > 0.7:
                        high_urgency_equips.append(equip)
                    elif urgency > 0.4:
                        medium_urgency_equips.append(equip)
                    else:
                        low_urgency_equips.append(equip)
                
                # Generar recomendaciones basadas en urgencia
                for equip in high_urgency_equips[:5]:
                    metrics = combined_metrics[equip]
                    if metrics.get('prediccion'):
                        expected_failures = round(metrics['prediccion']['expected_failures'], 1)
                        next_date = datetime.now() + timedelta(days=min(10, metrics['days_since_last_failure']))
                        insights['alertas_urgentes'].append(
                            f"ALERTA CRÍTICA: {equip} requiere mantenimiento INMEDIATO. "
                            f"Previsión: {expected_failures} fallos en los próximos 30 días. "
                            f"Programar inspección antes del {next_date.strftime('%d-%m-%Y')}"
                        )
                    else:
                        next_date = datetime.now() + timedelta(days=7)
                        insights['alertas_urgentes'].append(
                            f"ALERTA CRÍTICA: {equip} requiere mantenimiento INMEDIATO. "
                            f"Ha tenido {metrics['total_failures']} fallos totales. "
                            f"Programar inspección antes del {next_date.strftime('%d-%m-%Y')}"
                        )
                
                for equip in medium_urgency_equips[:8]:
                    metrics = combined_metrics[equip]
                    next_date = datetime.now() + timedelta(days=15)
                    insights['recomendaciones_predictivas'].append(
                        f"PRIORIDAD MEDIA: Programar mantenimiento para {equip} antes del {next_date.strftime('%d-%m-%Y')}. "
                        f"Último fallo hace {round(metrics['days_since_last_failure'])} días."
                    )
                
                for equip in low_urgency_equips[:5]:
                    metrics = combined_metrics[equip]
                    next_date = datetime.now() + timedelta(days=30)
                    insights['recomendaciones_preventivas'].append(
                        f"PREVENCIÓN: Incluir {equip} en plan de mantenimiento a 30 días. "
                        f"Verificar estado durante mantenimiento rutinario."
                    )
                
                # Añadir recomendaciones generales
                insights['recomendaciones_preventivas'].append(
                    f"Programar limpieza general de CDVs en estaciones con alta frecuencia de fallos cada 3 meses."
                )
                
                insights['recomendaciones_predictivas'].append(
                    f"Implementar revisión semanal de los {len(high_urgency_equips)} CDVs con mayor frecuencia de fallos."
                )
                
                # Analizar fallos de ocupación
                if 'fallos_ocupacion' in dfs and not dfs['fallos_ocupacion'].empty:
                    fo_df = dfs['fallos_ocupacion']
                    
                    # Agrupar por equipo para encontrar los más problemáticos
                    problematic_equip = fo_df.groupby('Equipo').size().sort_values(ascending=False)
                    
                    if not problematic_equip.empty:
                        # Top 5 equipos con más fallos
                        top_equipos = problematic_equip.head(5).index.tolist()
                        insights['resumen']['top_equipos_fallos_ocupacion'] = top_equipos
                        
                        # Detección de patrones temporales
                        if 'Fecha Hora' in fo_df.columns:
                            fo_df['hora'] = fo_df['Fecha Hora'].dt.hour
                            hour_distribution = fo_df['hora'].value_counts().sort_index()
                            
                            # Detectar horas pico de fallos
                            peak_hours = hour_distribution[hour_distribution > hour_distribution.mean() + hour_distribution.std()].index.tolist()
                            if peak_hours:
                                insights['patrones_detectados'].append(
                                    f"Se detectan más fallos de ocupación durante las horas: {', '.join(map(str, peak_hours))}"
                                )
                                insights['recomendaciones_predictivas'].append(
                                    f"Programar inspecciones adicionales durante las horas pico de fallos: {', '.join(map(str, peak_hours))}"
                                )
                
                # Analizar fallos de liberación
                if 'fallos_liberacion' in dfs and not dfs['fallos_liberacion'].empty:
                    fl_df = dfs['fallos_liberacion']
                    
                    # Agrupar por equipo para encontrar los más problemáticos
                    problematic_equip_fl = fl_df.groupby('Equipo').size().sort_values(ascending=False)
                    
                    if not problematic_equip_fl.empty:
                        # Top 5 equipos con más fallos de liberación
                        top_equipos_fl = problematic_equip_fl.head(5).index.tolist()
                        insights['resumen']['top_equipos_fallos_liberacion'] = top_equipos_fl
                
                # Análisis conjunto para equipos con ambos tipos de fallos
                if 'fallos_ocupacion' in dfs and 'fallos_liberacion' in dfs:
                    fo_equipos = set(dfs['fallos_ocupacion']['Equipo'].unique())
                    fl_equipos = set(dfs['fallos_liberacion']['Equipo'].unique())
                    
                    common_equipos = fo_equipos.intersection(fl_equipos)
                    if common_equipos:
                        insights['resumen']['equipos_con_ambos_fallos'] = list(common_equipos)
                        insights['recomendaciones_predictivas'].insert(0,
                            f"ALTA PRIORIDAD: Considerar reemplazo preventivo de los CDVs con ambos tipos de fallos: {', '.join(list(common_equipos)[:3])}"
                        )
                
                # Análisis de tendencias recientes
                if 'ocupaciones' in dfs and not dfs['ocupaciones'].empty:
                    ocup_df = dfs['ocupaciones']
                    
                    # Convertir Count a numérico si es string
                    if 'Count' in ocup_df.columns and ocup_df['Count'].dtype == 'object':
                        ocup_df['Count'] = pd.to_numeric(ocup_df['Count'], errors='coerce')
                    
                    # Analizar tendencias por día de la semana
                    if 'Fecha' in ocup_df.columns:
                        ocup_df['dia_semana'] = ocup_df['Fecha'].dt.day_name()
                        day_avg = ocup_df.groupby('dia_semana')['Count'].mean().sort_values(ascending=False)
                        
                        insights['resumen']['dia_mayor_ocupacion'] = day_avg.index[0] if not day_avg.empty else "No disponible"
                        insights['patrones_detectados'].append(
                            f"El día con mayor promedio de ocupaciones es {day_avg.index[0] if not day_avg.empty else 'No disponible'}"
                        )
            
            elif self.analysis_type == "ADV":
                # Implementación para ADV (similar a la de CDV pero adaptada)
                if 'discordancias' in dfs and not dfs['discordancias'].empty:
                    disc_df = dfs['discordancias']
                    
                    # Calcular métricas de confiabilidad
                    equip_col = 'Equipo Estacion' if 'Equipo Estacion' in disc_df.columns else 'Equipo'
                    if equip_col in disc_df.columns:
                        reliability_disc = self.calculate_equipment_reliability(
                            disc_df, 
                            equipment_col=equip_col, 
                            failure_date_col='Fecha Hora'
                        )
                        
                        insights['metricas_confiabilidad']['discordancias'] = reliability_disc
                        
                        # Generar recomendaciones basadas en urgencia
                        high_urgency = []
                        medium_urgency = []
                        low_urgency = []
                        
                        for equip, metrics in reliability_disc.items():
                            if metrics['maintenance_urgency'] > 0.7:
                                high_urgency.append(equip)
                            elif metrics['maintenance_urgency'] > 0.4:
                                medium_urgency.append(equip)
                            else:
                                low_urgency.append(equip)
                        
                        # Agregar recomendaciones
                        for equip in high_urgency[:3]:
                            metrics = reliability_disc[equip]
                            next_date = datetime.now() + timedelta(days=7)
                            insights['alertas_urgentes'].append(
                                f"ALERTA CRÍTICA: La aguja {equip} requiere mantenimiento URGENTE. "
                                f"Ha tenido {metrics['total_failures']} discordancias. "
                                f"Programar revisión antes del {next_date.strftime('%d-%m-%Y')}"
                            )
                        
                        for equip in medium_urgency[:5]:
                            metrics = reliability_disc[equip]
                            next_date = datetime.now() + timedelta(days=15)
                            insights['recomendaciones_predictivas'].append(
                                f"PRIORIDAD MEDIA: Programar lubricación para {equip} antes del {next_date.strftime('%d-%m-%Y')}. "
                                f"Última discordancia hace {round(metrics['days_since_last_failure'])} días."
                            )
                        
                        for equip in low_urgency[:5]:
                            metrics = reliability_disc[equip]
                            next_date = datetime.now() + timedelta(days=30)
                            insights['recomendaciones_preventivas'].append(
                                f"PREVENCIÓN: Incluir {equip} en plan de lubricación del próximo mes. "
                                f"Verificar durante mantenimiento rutinario."
                            )
                    
                    # Contar discordancias por equipo
                    if equip_col in disc_df.columns:
                        disc_count = disc_df[equip_col].value_counts().head(5)
                        top_disc_equipos = disc_count.index.tolist()
                        
                        insights['resumen']['top_equipos_discordancias'] = top_disc_equipos
                        insights['recomendaciones_predictivas'].append(
                            f"PREDICTIVO: Reprogramar parámetros de control para las agujas con mayor frecuencia de discordancias: {', '.join(top_disc_equipos[:3])}"
                        )
                
                if 'movimientos' in dfs and not dfs['movimientos'].empty:
                    mov_df = dfs['movimientos']
                    
                    # Convertir Count a numérico si es string
                    if 'Count' in mov_df.columns and mov_df['Count'].dtype == 'object':
                        mov_df['Count'] = pd.to_numeric(mov_df['Count'], errors='coerce')
                    
                    # Identificar agujas con mayor movimiento
                    if 'Equipo' in mov_df.columns and 'Count' in mov_df.columns:
                        mov_count = mov_df.groupby('Equipo')['Count'].sum().sort_values(ascending=False)
                        top_mov_equipos = mov_count.head(5).index.tolist()
                        
                        insights['resumen']['top_equipos_movimientos'] = top_mov_equipos
                        
                        # Generar recomendaciones de mantenimiento preventivo basado en frecuencia de uso
                        high_usage = []
                        medium_usage = []
                        
                        for equipo, count in mov_count.items():
                            if count > 100:
                                high_usage.append(equipo)
                            elif count > 50:
                                medium_usage.append(equipo)
                        
                        # Recomendaciones basadas en uso
                        for equip in high_usage[:3]:
                            insights['recomendaciones_predictivas'].append(
                                f"PREDICTIVO: Programar lubricación semanal para {equip} debido a su alto uso ({mov_count[equip]} movimientos)."
                            )
                        
                        for equip in medium_usage[:3]:
                            insights['recomendaciones_preventivas'].append(
                                f"PREVENCIÓN: Incluir {equip} en plan de lubricación quincenal."
                            )
                        
                        # Cruce con discordancias para identificar equipos críticos
                        if 'discordancias' in dfs and not dfs['discordancias'].empty:
                            equip_col = 'Equipo Estacion' if 'Equipo Estacion' in dfs['discordancias'].columns else 'Equipo'
                            disc_equipos = set(dfs['discordancias'][equip_col].unique())
                            high_usage_with_disc = set(high_usage).intersection(disc_equipos)
                            
                            if high_usage_with_disc:
                                insights['alertas_urgentes'].append(
                                    f"CRÍTICO: Las agujas {', '.join(list(high_usage_with_disc)[:3])} tienen alto uso Y discordancias. "
                                    f"Deben ser revisadas inmediatamente."
                                )
            
            # Generar recomendaciones generales basadas en el tipo de análisis
            if self.analysis_type == "CDV":
                insights['recomendaciones_preventivas'].append(
                    "Establecer un programa de inspección visual mensual para los CDVs con mayor frecuencia de fallos"
                )
                insights['recomendaciones_preventivas'].append(
                    "Implementar un protocolo de limpieza trimestral para los circuitos de vía en estaciones con mayor tráfico"
                )
            elif self.analysis_type == "ADV":
                insights['recomendaciones_preventivas'].append(
                    "Establecer un programa de inspección y lubricación preventiva para agujas con más de 50 movimientos diarios"
                )
                insights['recomendaciones_preventivas'].append(
                    "Verificar mensualmente la calibración de los sistemas de detección en agujas con discordancias recurrentes"
                )
            
            # Guardar insights para uso posterior
            self.insights = insights
            logger.info(f"Insights generados exitosamente para {self.line} - {self.analysis_type}")
            return insights
            
        except Exception as e:
            logger.error(f"Error al generar insights: {str(e)}")
            # Devolver insights básicos en caso de error
            self.insights = insights
            return insights
    
    def detect_anomalies(self, df, column, contamination=0.05):
        """Detectar anomalías utilizando Isolation Forest"""
        try:
            if df.empty or column not in df.columns:
                return pd.Series([False] * len(df))
            
            # Convertir a numérico si es necesario
            if df[column].dtype == 'object':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            
            # Preparar datos para el modelo
            X = df[column].values.reshape(-1, 1)
            
            # Aplicar Isolation Forest
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X)
            
            # -1 para anomalías, 1 para inliers
            return pd.Series(preds == -1, index=df.index)
            
        except Exception as e:
            logger.error(f"Error al detectar anomalías: {str(e)}")
            return pd.Series([False] * len(df))
        

    def calculate_equipment_reliability(self, df, equipment_col, failure_date_col=None, period_days=30):
        """Calcular métricas de confiabilidad por equipo"""
        reliability_data = {}
        
        try:
            if df is None or df.empty:
                return reliability_data
            
            # Agrupar por equipo
            equipment_counts = df[equipment_col].value_counts().reset_index()
            equipment_counts.columns = ['equipment', 'failures']
            
            # Si tenemos fechas, calcular métricas temporales
            if failure_date_col and failure_date_col in df.columns:
                now = datetime.now()
                for equipment in equipment_counts['equipment']:
                    equip_df = df[df[equipment_col] == equipment]
                    
                    # Ordenar por fecha
                    equip_df = equip_df.sort_values(by=failure_date_col)
                    
                    if len(equip_df) > 1:
                        # Calcular tiempo medio entre fallos (MTBF) en días
                        equip_df['next_failure'] = equip_df[failure_date_col].shift(-1)
                        equip_df['time_between_failures'] = (equip_df['next_failure'] - equip_df[failure_date_col]).dt.total_seconds() / (60*60*24)
                        mtbf = equip_df['time_between_failures'].mean()
                        
                        # Predecir próximo fallo basado en MTBF
                        last_failure = equip_df[failure_date_col].iloc[-1]
                        next_failure_prediction = last_failure + timedelta(days=mtbf if not pd.isna(mtbf) else 30)
                        
                        # Calcular tendencia (¿están aumentando los fallos?)
                        recent_failures = equip_df[equip_df[failure_date_col] >= (now - timedelta(days=period_days))]
                        is_increasing = len(recent_failures) > len(equip_df) / (365 / period_days)
                        
                        # Calcular días desde último fallo
                        days_since_last_failure = (now - last_failure).total_seconds() / (60*60*24)
                        
                        # Urgencia de mantenimiento
                        if mtbf:
                            maintenance_urgency = days_since_last_failure / mtbf
                        else:
                            maintenance_urgency = 0.5  # Valor por defecto
                        
                        reliability_data[equipment] = {
                            'mtbf': mtbf if not pd.isna(mtbf) else None,
                            'total_failures': len(equip_df),
                            'recent_failures': len(recent_failures),
                            'last_failure_date': last_failure,
                            'days_since_last_failure': days_since_last_failure,
                            'next_failure_prediction': next_failure_prediction,
                            'is_increasing': is_increasing,
                            'maintenance_urgency': min(1.0, maintenance_urgency)  # Normalizar a 1.0 max
                        }
                    else:
                        # Solo un fallo, no podemos calcular MTBF
                        last_failure = equip_df[failure_date_col].iloc[0]
                        days_since_last_failure = (now - last_failure).total_seconds() / (60*60*24)
                        
                        reliability_data[equipment] = {
                            'mtbf': None,
                            'total_failures': 1,
                            'recent_failures': 1 if (now - last_failure).days <= period_days else 0,
                            'last_failure_date': last_failure,
                            'days_since_last_failure': days_since_last_failure,
                            'next_failure_prediction': None,
                            'is_increasing': False,
                            'maintenance_urgency': 0.3  # Valor por defecto para un solo fallo
                        }
            
            return reliability_data
        
        except Exception as e:
            logger.error(f"Error calculando métricas de confiabilidad: {str(e)}")
            return reliability_data    
        

    def predict_failure_trend(self, df, equipment_col, date_col, equipment, days_forecast=30):
        """Predecir tendencia de fallos para un equipamiento específico"""
        try:
            if df is None or df.empty or equipment_col not in df.columns or date_col not in df.columns:
                return None
            
            equip_df = df[df[equipment_col] == equipment].copy()
            if len(equip_df) < 5:  # Necesitamos suficientes datos para una predicción
                return None
            
            # Agrupar por día para contar fallos
            equip_df['date'] = pd.to_datetime(equip_df[date_col]).dt.date
            daily_failures = equip_df.groupby('date').size().reset_index(name='failures')
            daily_failures['date'] = pd.to_datetime(daily_failures['date'])
            
            # Crear serie temporal completa (incluyendo días sin fallos)
            date_range = pd.date_range(start=daily_failures['date'].min(), end=daily_failures['date'].max())
            ts = pd.DataFrame({'date': date_range})
            ts = ts.merge(daily_failures, on='date', how='left')
            ts['failures'] = ts['failures'].fillna(0)
            
            # Intentar ajustar un modelo ARIMA simple
            try:
                # Preparar datos para ARIMA
                ts_data = ts['failures'].values
                model = ARIMA(ts_data, order=(1,0,1))
                model_fit = model.fit()
                
                # Predecir
                forecast = model_fit.forecast(steps=days_forecast)
                return {
                    'forecasted_values': forecast.tolist(),
                    'forecasted_dates': [(ts['date'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days_forecast)],
                    'expected_failures': sum(forecast)
                }
            except:
                # Si falla ARIMA, usar regresión lineal simple
                X = np.array(range(len(ts))).reshape(-1, 1)
                y = ts['failures'].values
                model = LinearRegression()
                model.fit(X, y)
                
                # Predecir
                X_future = np.array(range(len(ts), len(ts) + days_forecast)).reshape(-1, 1)
                forecast = model.predict(X_future)
                forecast = np.maximum(forecast, 0)  # No permitir valores negativos
                
                return {
                    'forecasted_values': forecast.tolist(),
                    'forecasted_dates': [(ts['date'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days_forecast)],
                    'expected_failures': sum(forecast)
                }
        
        except Exception as e:
            logger.error(f"Error en predicción de tendencia: {str(e)}")
            return None

    
    def create_dashboard(self):
        """Crear y configurar el dashboard web"""
        if not self.dataframes:
            logger.error("No hay datos cargados para generar el dashboard")
            return False
        
        try:
            # Generar insights iniciales si no existen
            if not self.insights:
                self.insights = self.generate_insights()
            
            # Crear aplicación Dash con estilos externos
            self.app = dash.Dash(
                __name__, 
                suppress_callback_exceptions=True,
                external_stylesheets=[
                    'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'
                ]
            )
            
            # Definir el layout del dashboard
            self.app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'fontFamily': 'Roboto, sans-serif'}, children=[
                # Header
                html.Div(style={'backgroundColor': '#2C3E50', 'color': 'white', 'padding': '20px', 'marginBottom': '20px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}, children=[
                    html.H2(f"Dashboard de Análisis - {self.line} {self.analysis_type}", 
                            style={'textAlign': 'center', 'fontWeight': 'bold'}),
                    html.P(f"Fecha de generación: {datetime.now().strftime('%d-%m-%Y %H:%M')}", 
                        style={'textAlign': 'center', 'opacity': '0.8'})
                ]),
                
                # Contenedor principal
                html.Div(className='container-fluid px-4', children=[
                    # Fila de KPIs
                    html.Div(className='row mb-4 g-3', children=self.create_kpi_cards()),  # Corrección aquí
                    
                    # Fila de gráficos principales
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-6', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Tendencia Temporal", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='time-trend-graph', figure=self.create_time_trend_figure())
                                ])
                            ])
                        ]),
                        html.Div(className='col-md-6', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Distribución por Equipo", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='equipment-distribution', figure=self.create_equipment_distribution_figure())
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila de filtros y controles
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Filtros y Controles", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    html.Div(className='row', children=[
                                        html.Div(className='col-md-4', children=[
                                            html.Label("Rango de Fechas:"),
                                            dcc.DatePickerRange(
                                                id='date-range',
                                                min_date_allowed=self.get_min_date(),
                                                max_date_allowed=self.get_max_date(),
                                                start_date=self.get_min_date(),
                                                end_date=self.get_max_date()
                                            ),
                                        ]),
                                        html.Div(className='col-md-4', children=[
                                            html.Label("Equipo:"),
                                            dcc.Dropdown(
                                                id='equipment-filter',
                                                options=[{'label': equipo, 'value': equipo} for equipo in self.get_equipment_list()],
                                                multi=True,
                                                placeholder="Seleccionar equipos..."
                                            ),
                                        ]),
                                        html.Div(className='col-md-4', children=[
                                            html.Label("Tipo de Visualización:"),
                                            dcc.RadioItems(
                                                id='visualization-type',
                                                options=[
                                                    {'label': 'Diario', 'value': 'daily'},
                                                    {'label': 'Semanal', 'value': 'weekly'},
                                                    {'label': 'Mensual', 'value': 'monthly'}
                                                ],
                                                value='daily',
                                                labelStyle={'display': 'block'}
                                            ),
                                        ])
                                    ]),
                                    html.Div(className='row mt-3', children=[
                                        html.Div(className='col-md-12 text-center', children=[
                                            html.Button('Aplicar filtros', id='apply-filters-button', className='btn btn-primary', style={
                                                'backgroundColor': self.colors['secondary'],
                                                'color': 'white',
                                                'fontWeight': 'bold',
                                                'padding': '10px 20px',
                                                'border': 'none',
                                                'borderRadius': '5px',
                                                'cursor': 'pointer'
                                            }),
                                        ])
                                    ])
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila de gráficos adicionales
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-6', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Distribución Horaria", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='hourly-distribution', figure=self.create_hourly_distribution_figure())
                                ])
                            ])
                        ]),
                        html.Div(className='col-md-6', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Mapa de Calor", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='heatmap', figure=self.create_heatmap_figure())
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila de alertas urgentes (nueva)
                    html.Div(id='alertas-container', className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header bg-danger text-white', children=[
                                    html.H5("Alertas Urgentes de Mantenimiento", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    html.Ul(id='alertas-list', children=[
                                        html.Li(alerta, className='alert alert-danger') 
                                        for alerta in self.insights.get('alertas_urgentes', [])
                                    ]) if self.insights.get('alertas_urgentes', []) else 
                                    html.P("No hay alertas urgentes en este momento", className='text-success')
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila de recomendaciones y análisis
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-6', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header bg-primary text-white', children=[
                                    html.H5("Mantenimiento Predictivo", className='card-title')
                                ]),
                                html.Div(className='card-body', id='maintenance-predictive-body', children=[
                                    html.Ul(id='recomendaciones-predictivas-list', children=[
                                        html.Li(rec, className='mb-2') 
                                        for rec in self.insights.get('recomendaciones_predictivas', [])
                                    ])
                                ])
                            ])
                        ]),
                        html.Div(className='col-md-6', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header bg-success text-white', children=[
                                    html.H5("Mantenimiento Preventivo", className='card-title')
                                ]),
                                html.Div(className='card-body', id='maintenance-preventive-body', children=[
                                    html.Ul(id='recomendaciones-preventivas-list', children=[
                                        html.Li(rec, className='mb-2') 
                                        for rec in self.insights.get('recomendaciones_preventivas', [])
                                    ])
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila de patrones detectados
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header bg-warning', children=[
                                    html.H5("Patrones Detectados", className='card-title')
                                ]),
                                html.Div(className='card-body', id='patrones-body', children=[
                                    html.Ul(id='patrones-list', children=[
                                        html.Li(pat) for pat in self.insights.get('patrones_detectados', [])
                                    ]) if self.insights.get('patrones_detectados', []) else 
                                    html.P("No se detectaron patrones significativos")
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila para tabla de datos
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Datos Detallados", className='card-title')
                                ]),
                                html.Div(className='card-body', style={'overflowX': 'auto'}, children=[
                                    self.create_data_table()
                                ])
                            ])
                        ])
                    ])
                ])
            ])
            
            # Configurar callbacks
            self.setup_callbacks()
            
            logger.info(f"Dashboard creado exitosamente para {self.line} - {self.analysis_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error al crear el dashboard: {str(e)}")
            return False
    
    def create_kpi_cards(self):
        """Crear tarjetas de KPI según el tipo de análisis"""
        kpi_cards = []
        
        try:
            if self.analysis_type == "CDV":
                # KPIs para CDV
                
                # Total de fallos de ocupación
                total_fo = len(self.dataframes.get('fallos_ocupacion', pd.DataFrame()))
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#E74C3C', 'borderBottom': 'none'}, 
                                    children=[html.H5("Fallos de Ocupación", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{total_fo}", className='display-4 mb-0 fw-bold'),
                                html.P("Total de fallos detectados", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Total de fallos de liberación
                total_fl = len(self.dataframes.get('fallos_liberacion', pd.DataFrame()))
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#F39C12', 'borderBottom': 'none'}, 
                                    children=[html.H5("Fallos de Liberación", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{total_fl}", className='display-4 mb-0 fw-bold'),
                                html.P("Total de fallos detectados", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Equipos que requieren atención urgente
                num_urgentes = 0
                if hasattr(self, 'insights') and self.insights and 'alertas_urgentes' in self.insights:
                    num_urgentes = len(self.insights['alertas_urgentes'])
                
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#C0392B', 'borderBottom': 'none'}, 
                                    children=[html.H5("Equipos Críticos", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{num_urgentes}", className='display-4 mb-0 fw-bold'),
                                html.P("Requieren atención inmediata", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Índice MTBF (Mean Time Between Failures) promedio
                mtbf_value = "N/A"
                try:
                    if total_fo > 0 and 'fallos_ocupacion' in self.dataframes and 'Fecha Hora' in self.dataframes['fallos_ocupacion'].columns:
                        # Calcular MTBF solo si tenemos datos de fechas
                        fo_df = self.dataframes['fallos_ocupacion'].copy()
                        fo_df['Fecha Hora'] = pd.to_datetime(fo_df['Fecha Hora'], errors='coerce')
                        
                        # Agrupar por equipo
                        mtbf_by_equip = {}
                        for equip in fo_df['Equipo'].unique():
                            eq_df = fo_df[fo_df['Equipo'] == equip].sort_values('Fecha Hora')
                            if len(eq_df) > 1:
                                # Calcular diferencias entre fechas consecutivas
                                eq_df['next_failure'] = eq_df['Fecha Hora'].shift(-1)
                                eq_df['time_diff'] = (eq_df['next_failure'] - eq_df['Fecha Hora']).dt.total_seconds() / (3600 * 24)  # en días
                                avg_mtbf = eq_df['time_diff'].mean()
                                if not pd.isna(avg_mtbf):
                                    mtbf_by_equip[equip] = avg_mtbf
                        
                        if mtbf_by_equip:
                            mtbf_value = f"{np.mean(list(mtbf_by_equip.values())):.1f} días"
                except Exception as e:
                    logger.error(f"Error calculando MTBF: {str(e)}")
                    mtbf_value = "Error de cálculo"
                
                # Tarjeta con MTBF  
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#3498DB', 'borderBottom': 'none'}, 
                                    children=[html.H5("Tiempo Medio Entre Fallos", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{mtbf_value}", className='display-4 mb-0 fw-bold'),
                                html.P("MTBF promedio del sistema", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Índice de fiabilidad
                fiabilidad = "N/A"
                fiabilidad_color = '#2ECC71'  # Color por defecto verde
                try:
                    ocupaciones_df = self.dataframes.get('ocupaciones', pd.DataFrame())
                    if not ocupaciones_df.empty and 'Count' in ocupaciones_df.columns:
                        # Convertir a numérico si es string
                        if ocupaciones_df['Count'].dtype == 'object':
                            ocupaciones_df['Count'] = pd.to_numeric(ocupaciones_df['Count'], errors='coerce')
                        
                        total_ocupaciones = ocupaciones_df['Count'].sum()
                        if total_ocupaciones > 0:
                            fiabilidad = 100 * (1 - (total_fo + total_fl) / total_ocupaciones)
                            fiabilidad = max(0, min(100, fiabilidad))  # Limitar entre 0 y 100
                            
                            # Asignar color según el valor
                            if fiabilidad < 70:
                                fiabilidad_color = '#E74C3C'  # Rojo
                            elif fiabilidad < 85:
                                fiabilidad_color = '#F39C12'  # Amarillo
                            elif fiabilidad < 95:
                                fiabilidad_color = '#3498DB'  # Azul
                except Exception as e:
                    logger.error(f"Error calculando fiabilidad: {str(e)}")
                    fiabilidad = "Error de cálculo"
                    fiabilidad_color = '#E74C3C'  # Rojo para error
                
                # Confiabilidad del sistema
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': fiabilidad_color, 'borderBottom': 'none'}, 
                                    children=[html.H5("Índice de Confiabilidad", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{fiabilidad if isinstance(fiabilidad, str) else f'{fiabilidad:.2f}%'}", 
                                    className='display-4 mb-0 fw-bold',
                                    style={'color': fiabilidad_color}),
                                html.P("Operaciones sin fallos", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Estado del sistema
                system_status = "ESTADO DESCONOCIDO"
                status_class = "#6c757d"  # Gris por defecto
                icon = "⚠️"
                
                if isinstance(fiabilidad, float):
                    if fiabilidad >= 95:
                        system_status = "ÓPTIMO"
                        status_class = "#2ECC71"  # Verde
                        icon = "✅"
                    elif fiabilidad >= 85:
                        system_status = "ACEPTABLE"
                        status_class = "#3498DB"  # Azul
                        icon = "✓"
                    elif fiabilidad >= 70:
                        system_status = "REQUIERE ATENCIÓN"
                        status_class = "#F39C12"  # Amarillo
                        icon = "⚠️"
                    else:
                        system_status = "CRÍTICO"
                        status_class = "#E74C3C"  # Rojo
                        icon = "❌"
                
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': status_class, 'borderBottom': 'none'}, 
                                    children=[html.H5("Estado del Sistema", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.Div([
                                    html.Span(icon, style={'fontSize': '2rem', 'marginRight': '10px'}),
                                    html.H2(f"{system_status}", 
                                        className='mb-0 fw-bold d-inline',
                                        style={'color': status_class})
                                ]),
                                html.P("Evaluación general", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
            # [El resto del código para ADV...]

        except Exception as e:
            logger.error(f"Error al crear tarjetas KPI: {str(e)}")
            # Tarjeta de error
            kpi_cards.append(
                html.Div(className='col-12', children=[
                    html.Div(className='card border-0 shadow-sm', style={'borderRadius': '10px'}, children=[
                        html.Div(className='card-header text-white text-center py-3', 
                                style={'backgroundColor': '#E74C3C', 'borderBottom': 'none'}, 
                                children=[html.H5("Error", className='m-0 fw-bold')]),
                        html.Div(className='card-body text-center', children=[
                            html.H5("Error al generar KPIs", className='card-title'),
                            html.P(f"Detalles: {str(e)}", className='card-text')
                        ])
                    ])
                ])
            )
        
        # Asegurar que siempre haya al menos una tarjeta
        if not kpi_cards:
            kpi_cards.append(
                html.Div(className='col-12', children=[
                    html.Div(className='card border-0 shadow-sm', style={'borderRadius': '10px'}, children=[
                        html.Div(className='card-header text-white text-center py-3', 
                                style={'backgroundColor': '#6c757d', 'borderBottom': 'none'}, 
                                children=[html.H5("Información", className='m-0 fw-bold')]),
                        html.Div(className='card-body text-center', children=[
                            html.H5("No hay datos suficientes", className='card-title'),
                            html.P("No se pudieron generar los indicadores de rendimiento", className='card-text')
                        ])
                    ])
                ])
            )
        
        return kpi_cards
    
    def create_time_trend_figure(self, dataframes=None):
        """Crear gráfico de tendencia temporal"""
        # Usar los dataframes filtrados si se proporcionan, o los originales si no
        dfs = dataframes if dataframes else self.dataframes
        
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in dfs and 'Fecha Hora' in dfs['fallos_ocupacion'].columns:
                df = dfs['fallos_ocupacion'].copy()
                
                # Agrupar por fecha para contar fallos
                df['fecha'] = pd.to_datetime(df['Fecha Hora']).dt.date
                fallas_por_dia = df.groupby('fecha').size().reset_index(name='conteo')
                fallas_por_dia['fecha'] = pd.to_datetime(fallas_por_dia['fecha'])
                
                fig = px.line(
                    fallas_por_dia, 
                    x='fecha', 
                    y='conteo',
                    labels={'fecha': 'Fecha', 'conteo': 'Número de Fallos'},
                    title="Tendencia de Fallos de Ocupación",
                    color_discrete_sequence=[self.colors['danger']]
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color=self.colors['text'],
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                return fig
            
        elif self.analysis_type == "ADV":
            if 'discordancias' in self.dataframes and 'Fecha Hora' in self.dataframes['discordancias'].columns:
                df = self.dataframes['discordancias'].copy()
                
                # Agrupar por fecha para contar discordancias
                df['fecha'] = pd.to_datetime(df['Fecha Hora']).dt.date
                disc_por_dia = df.groupby('fecha').size().reset_index(name='conteo')
                disc_por_dia['fecha'] = pd.to_datetime(disc_por_dia['fecha'])
                
                fig = px.line(
                    disc_por_dia, 
                    x='fecha', 
                    y='conteo',
                    labels={'fecha': 'Fecha', 'conteo': 'Número de Discordancias'},
                    title="Tendencia de Discordancias en Agujas",
                    color_discrete_sequence=[self.colors['danger']]
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color=self.colors['text'],
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                return fig
        
        # Figura vacía en caso de no tener datos
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos disponibles para mostrar tendencia temporal",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="Valor"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text']
        )
        
        return fig
    
    def create_equipment_distribution_figure(self, dataframes=None):
        """Crear gráfico de distribución por equipo"""
        # Usar los dataframes filtrados si se proporcionan, o los originales si no
        dfs = dataframes if dataframes else self.dataframes
        
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in dfs and 'Equipo' in dfs['fallos_ocupacion'].columns:
                df = dfs['fallos_ocupacion'].copy()
                
                # Contar fallos por equipo
                fallas_por_equipo = df['Equipo'].value_counts().reset_index()
                fallas_por_equipo.columns = ['Equipo', 'Conteo']
                
                # Tomar los 15 equipos con más fallos
                top_equipos = fallas_por_equipo.head(15)
                
                fig = px.bar(
                    top_equipos, 
                    x='Equipo', 
                    y='Conteo',
                    labels={'Equipo': 'CDV', 'Conteo': 'Número de Fallos'},
                    title="Distribución de Fallos por CDV (Top 15)",
                    color_discrete_sequence=[self.colors['primary']]
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color=self.colors['text'],
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis={'categoryorder':'total descending'}
                )
                
                return fig
                
        elif self.analysis_type == "ADV":
            if 'discordancias' in dfs and 'Equipo Estacion' in dfs['discordancias'].columns:
                df = dfs['discordancias'].copy()
                
                # Contar discordancias por equipo
                disc_por_equipo = df['Equipo Estacion'].value_counts().reset_index()
                disc_por_equipo.columns = ['Equipo', 'Conteo']
                
                # Tomar los 15 equipos con más discordancias
                top_equipos = disc_por_equipo.head(15)
                
                fig = px.bar(
                    top_equipos, 
                    x='Equipo', 
                    y='Conteo',
                    labels={'Equipo': 'Aguja', 'Conteo': 'Número de Discordancias'},
                    title="Distribución de Discordancias por Aguja (Top 15)",
                    color_discrete_sequence=[self.colors['primary']]
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color=self.colors['text'],
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis={'categoryorder':'total descending'}
                )
                
                return fig
        
        # Figura vacía en caso de no tener datos
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos disponibles para mostrar distribución por equipo",
            xaxis=dict(title="Equipo"),
            yaxis=dict(title="Conteo"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text']
        )
        
        return fig
    
    def create_hourly_distribution_figure(self, dataframes=None, viz_type='daily'):
        """Crear gráfico de distribución horaria"""
        # Usar los dataframes filtrados si se proporcionan, o los originales si no
        dfs = dataframes if dataframes else self.dataframes
        
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in dfs and 'Fecha Hora' in dfs['fallos_ocupacion'].columns:
                df = dfs['fallos_ocupacion'].copy()
                
                # Extraer hora del día
                df['hora'] = pd.to_datetime(df['Fecha Hora']).dt.hour
                
                # Aplicar agrupación basada en el tipo de visualización
                if viz_type == 'weekly':
                    df['dia_semana'] = pd.to_datetime(df['Fecha Hora']).dt.day_name()
                    fallas_por_tiempo = df.groupby('dia_semana').size().reset_index(name='Conteo')
                    fallas_por_tiempo.columns = ['Periodo', 'Conteo']
                    titulo = "Distribución de Fallos por Día de la Semana"
                    x_label = "Día de la Semana"
                elif viz_type == 'monthly':
                    df['mes'] = pd.to_datetime(df['Fecha Hora']).dt.month_name()
                    fallas_por_tiempo = df.groupby('mes').size().reset_index(name='Conteo')
                    fallas_por_tiempo.columns = ['Periodo', 'Conteo']
                    titulo = "Distribución de Fallos por Mes"
                    x_label = "Mes"
                else:  # default: daily
                    fallas_por_tiempo = df['hora'].value_counts().reset_index()
                    fallas_por_tiempo.columns = ['Periodo', 'Conteo']
                    fallas_por_tiempo = fallas_por_tiempo.sort_values('Periodo')
                    titulo = "Distribución Horaria de Fallos"
                    x_label = "Hora del Día"
                
                fig = px.line(
                    fallas_por_tiempo, 
                    x='Periodo', 
                    y='Conteo',
                    labels={'Periodo': x_label, 'Conteo': 'Número de Fallos'},
                    title=titulo,
                    markers=True,
                    color_discrete_sequence=[self.colors['info']]
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color=self.colors['text'],
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                if viz_type == 'daily':
                    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
                
                return fig
                
        elif self.analysis_type == "ADV":
            if 'discordancias' in dfs and 'Fecha Hora' in dfs['discordancias'].columns:
                df = dfs['discordancias'].copy()
                
                # Extraer hora del día
                df['hora'] = pd.to_datetime(df['Fecha Hora']).dt.hour
                
                # Aplicar agrupación basada en el tipo de visualización
                if viz_type == 'weekly':
                    df['dia_semana'] = pd.to_datetime(df['Fecha Hora']).dt.day_name()
                    disc_por_tiempo = df.groupby('dia_semana').size().reset_index(name='Conteo')
                    disc_por_tiempo.columns = ['Periodo', 'Conteo']
                    titulo = "Distribución de Discordancias por Día de la Semana"
                    x_label = "Día de la Semana"
                elif viz_type == 'monthly':
                    df['mes'] = pd.to_datetime(df['Fecha Hora']).dt.month_name()
                    disc_por_tiempo = df.groupby('mes').size().reset_index(name='Conteo')
                    disc_por_tiempo.columns = ['Periodo', 'Conteo']
                    titulo = "Distribución de Discordancias por Mes"
                    x_label = "Mes"
                else:  # default: daily
                    disc_por_hora = df['hora'].value_counts().reset_index()
                    disc_por_hora.columns = ['Periodo', 'Conteo']
                    disc_por_tiempo = disc_por_hora.sort_values('Periodo')
                    titulo = "Distribución Horaria de Discordancias"
                    x_label = "Hora del Día"
                
                fig = px.line(
                    disc_por_tiempo, 
                    x='Periodo', 
                    y='Conteo',
                    labels={'Periodo': x_label, 'Conteo': 'Número de Discordancias'},
                    title=titulo,
                    markers=True,
                    color_discrete_sequence=[self.colors['info']]
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_color=self.colors['text'],
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                if viz_type == 'daily':
                    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
                
                return fig
        
        # Figura vacía en caso de no tener datos
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos disponibles para mostrar distribución horaria",
            xaxis=dict(title="Hora"),
            yaxis=dict(title="Conteo"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text']
        )
        
        return fig
    
    def create_heatmap_figure(self, dataframes=None, viz_type='daily'):
        """Crear mapa de calor (día de la semana vs. hora)"""
        # Usar los dataframes filtrados si se proporcionan, o los originales si no
        dfs = dataframes if dataframes else self.dataframes
        
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in dfs and 'Fecha Hora' in dfs['fallos_ocupacion'].columns:
                df = dfs['fallos_ocupacion'].copy()
                
                # Extraer diferentes dimensiones temporales
                fecha_dt = pd.to_datetime(df['Fecha Hora'])
                df['dia_semana'] = fecha_dt.dt.day_name()
                df['hora'] = fecha_dt.dt.hour
                df['mes'] = fecha_dt.dt.month_name()
                df['semana'] = fecha_dt.dt.isocalendar().week
                
                # Configurar dimensiones basadas en el tipo de visualización
                if viz_type == 'monthly':
                    index_col = 'mes'
                    columns_col = 'dia_semana'
                    title = "Mapa de Calor: Fallos por Mes y Día de la Semana"
                    x_label = "Día de la Semana"
                    y_label = "Mes"
                elif viz_type == 'weekly':
                    index_col = 'semana'
                    columns_col = 'dia_semana'
                    title = "Mapa de Calor: Fallos por Semana y Día"
                    x_label = "Día de la Semana"
                    y_label = "Semana del Año"
                else:  # default: daily
                    index_col = 'dia_semana'
                    columns_col = 'hora'
                    title = "Mapa de Calor: Fallos por Día y Hora"
                    x_label = "Hora del Día"
                    y_label = "Día de la Semana"
                
                # Orden de los días
                dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Crear tabla pivote para el heatmap
                try:
                    heatmap_data = pd.pivot_table(
                        df, 
                        values='Equipo',
                        index=index_col,
                        columns=columns_col,
                        aggfunc='count',
                        fill_value=0
                    )
                    
                    # Reordenar días si es necesario
                    if columns_col == 'dia_semana':
                        heatmap_data = heatmap_data.reindex(columns=dias_orden)
                    elif index_col == 'dia_semana':
                        heatmap_data = heatmap_data.reindex(dias_orden)
                    
                    # Crear heatmap
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x=x_label, y=y_label, color="Número de Fallos"),
                        title=title,
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color=self.colors['text'],
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    
                    return fig
                except:
                    # En caso de error con pivot_table (normalmente por falta de datos)
                    pass
                
        elif self.analysis_type == "ADV":
            if 'discordancias' in dfs and 'Fecha Hora' in dfs['discordancias'].columns:
                df = dfs['discordancias'].copy()
                
                # Extraer diferentes dimensiones temporales
                fecha_dt = pd.to_datetime(df['Fecha Hora'])
                df['dia_semana'] = fecha_dt.dt.day_name()
                df['hora'] = fecha_dt.dt.hour
                df['mes'] = fecha_dt.dt.month_name()
                df['semana'] = fecha_dt.dt.isocalendar().week
                
                # Configurar dimensiones basadas en el tipo de visualización
                if viz_type == 'monthly':
                    index_col = 'mes'
                    columns_col = 'dia_semana'
                    title = "Mapa de Calor: Discordancias por Mes y Día de la Semana"
                    x_label = "Día de la Semana"
                    y_label = "Mes"
                elif viz_type == 'weekly':
                    index_col = 'semana'
                    columns_col = 'dia_semana'
                    title = "Mapa de Calor: Discordancias por Semana y Día"
                    x_label = "Día de la Semana"
                    y_label = "Semana del Año"
                else:  # default: daily
                    index_col = 'dia_semana'
                    columns_col = 'hora'
                    title = "Mapa de Calor: Discordancias por Día y Hora"
                    x_label = "Hora del Día"
                    y_label = "Día de la Semana"
                
                # Orden de los días
                dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Crear tabla pivote para el heatmap
                try:
                    heatmap_data = pd.pivot_table(
                        df, 
                        values='Equipo Estacion',
                        index=index_col,
                        columns=columns_col,
                        aggfunc='count',
                        fill_value=0
                    )
                    
                    # Reordenar días si es necesario
                    if columns_col == 'dia_semana':
                        heatmap_data = heatmap_data.reindex(columns=dias_orden)
                    elif index_col == 'dia_semana':
                        heatmap_data = heatmap_data.reindex(dias_orden)
                    
                    # Crear heatmap
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x=x_label, y=y_label, color="Número de Discordancias"),
                        title=title,
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color=self.colors['text'],
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    
                    return fig
                except:
                    # En caso de error con pivot_table (normalmente por falta de datos)
                    pass
        
        # Figura vacía en caso de no tener datos
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos disponibles para mostrar mapa de calor",
            xaxis=dict(title="Hora"),
            yaxis=dict(title="Día"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text']
        )
        
        return fig
    
    def create_data_table(self):
        """Crear tabla de datos"""
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in self.dataframes and not self.dataframes['fallos_ocupacion'].empty:
                df = self.dataframes['fallos_ocupacion'].copy()
                
                # Seleccionar columnas relevantes
                if 'Fecha Hora' in df.columns and 'Equipo' in df.columns and 'Estacion' in df.columns:
                    df = df[['Fecha Hora', 'Equipo', 'Estacion', 'Diff.Time_+1_row']]
                    
                    # Formatear para mostrar
                    df['Fecha Hora'] = df['Fecha Hora'].dt.strftime('%d-%m-%Y %H:%M:%S')
                    df['Diff.Time_+1_row'] = df['Diff.Time_+1_row'].astype(str)
                    
                    return html.Table(
                        # Encabezado
                        [html.Tr([html.Th(col) for col in df.columns])] +
                        # Cuerpo
                        [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(50, len(df)))],
                        className='table table-striped table-hover'
                    )
                
        elif self.analysis_type == "ADV":
            if 'discordancias' in self.dataframes and not self.dataframes['discordancias'].empty:
                df = self.dataframes['discordancias'].copy()
                
                # Seleccionar columnas relevantes
                columns_to_show = ['Fecha Hora', 'Equipo Estacion', 'Linea']
                columns_present = [col for col in columns_to_show if col in df.columns]
                
                if columns_present:
                    df = df[columns_present]
                    
                    # Formatear para mostrar si es necesario
                    if 'Fecha Hora' in df.columns:
                        df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora']).dt.strftime('%d-%m-%Y %H:%M:%S')
                    
                    return html.Table(
                        # Encabezado
                        [html.Tr([html.Th(col) for col in df.columns])] +
                        # Cuerpo
                        [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(50, len(df)))],
                        className='table table-striped table-hover'
                    )
        
        # Tabla vacía en caso de no tener datos
        return html.Div("No hay datos disponibles para mostrar en la tabla.")
    
    def get_min_date(self):
        """Obtener la fecha mínima de los datos"""
        min_date = datetime.now()
        
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in self.dataframes and 'Fecha Hora' in self.dataframes['fallos_ocupacion'].columns:
                date_min = self.dataframes['fallos_ocupacion']['Fecha Hora'].min()
                if date_min and not pd.isna(date_min):
                    min_date = min(min_date, date_min)
            
            if 'fallos_liberacion' in self.dataframes and 'Fecha Hora' in self.dataframes['fallos_liberacion'].columns:
                date_min = self.dataframes['fallos_liberacion']['Fecha Hora'].min()
                if date_min and not pd.isna(date_min):
                    min_date = min(min_date, date_min)
        
        elif self.analysis_type == "ADV":
            if 'discordancias' in self.dataframes and 'Fecha Hora' in self.dataframes['discordancias'].columns:
                date_min = pd.to_datetime(self.dataframes['discordancias']['Fecha Hora']).min()
                if date_min and not pd.isna(date_min):
                    min_date = min(min_date, date_min)
        
        # Retroceder 30 días por defecto
        return min_date.date()
    
    def get_max_date(self):
        """Obtener la fecha máxima de los datos"""
        max_date = datetime.now().date()
        return max_date
    
    def get_equipment_list(self):
        """Obtener lista de equipos disponibles"""
        equipos = []
        
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' in self.dataframes and 'Equipo' in self.dataframes['fallos_ocupacion'].columns:
                equipos.extend(self.dataframes['fallos_ocupacion']['Equipo'].unique())
            
            if 'fallos_liberacion' in self.dataframes and 'Equipo' in self.dataframes['fallos_liberacion'].columns:
                equipos.extend(self.dataframes['fallos_liberacion']['Equipo'].unique())
        
        elif self.analysis_type == "ADV":
            if 'discordancias' in self.dataframes and 'Equipo Estacion' in self.dataframes['discordancias'].columns:
                equipos.extend(self.dataframes['discordancias']['Equipo Estacion'].unique())
        
        return sorted(list(set(equipos)))
    
    def setup_callbacks(self):
        """Configurar callbacks para interactividad"""
        if not self.app:
            return
        
        # Definir la función del callback
        def update_graphs_and_recommendations(n_clicks, start_date, end_date, selected_equipments, viz_type):
            # No actualizar si no se ha presionado el botón
            if n_clicks is None:
                # Retornar valores iniciales
                return [
                    self.create_time_trend_figure(), 
                    self.create_equipment_distribution_figure(),
                    self.create_hourly_distribution_figure(),
                    self.create_heatmap_figure(),
                    # Lista de elementos para las alertas
                    [html.Li(alerta, className='alert alert-danger') for alerta in self.insights.get('alertas_urgentes', [])] 
                    if self.insights.get('alertas_urgentes', []) else [html.P("No hay alertas urgentes en este momento", className='text-success')],
                    # Lista para recomendaciones predictivas
                    [html.Li(rec, className='mb-2') for rec in self.insights.get('recomendaciones_predictivas', [])],
                    # Lista para recomendaciones preventivas
                    [html.Li(rec, className='mb-2') for rec in self.insights.get('recomendaciones_preventivas', [])],
                    # Lista para patrones detectados
                    [html.Li(pat) for pat in self.insights.get('patrones_detectados', [])]
                    if self.insights.get('patrones_detectados', []) else [html.P("No se detectaron patrones significativos")]
                ]
                
            try:
                # Preparar fecha de inicio y fin
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Crear copias filtradas de los dataframes originales
                filtered_dfs = {}
                
                # Aplicar filtros a todos los dataframes
                for key, df in self.dataframes.items():
                    if df is not None and not df.empty:
                        filtered_dfs[key] = df.copy()
                        
                        # Filtrar por fecha si la columna adecuada existe
                        if 'Fecha Hora' in df.columns and start_date and end_date:
                            # Asegurar que Fecha Hora es datetime
                            filtered_dfs[key]['Fecha Hora'] = pd.to_datetime(filtered_dfs[key]['Fecha Hora'])
                            filtered_dfs[key] = filtered_dfs[key][
                                (filtered_dfs[key]['Fecha Hora'] >= start_date) & 
                                (filtered_dfs[key]['Fecha Hora'] <= end_date)
                            ]
                        elif 'Fecha' in df.columns and start_date and end_date:
                            # Asegurar que Fecha es datetime
                            filtered_dfs[key]['Fecha'] = pd.to_datetime(filtered_dfs[key]['Fecha'])
                            filtered_dfs[key] = filtered_dfs[key][
                                (filtered_dfs[key]['Fecha'] >= pd.to_datetime(start_date)) & 
                                (filtered_dfs[key]['Fecha'] <= pd.to_datetime(end_date))
                            ]
                        
                        # Filtrar por equipamiento según el tipo de análisis
                        if selected_equipments and len(selected_equipments) > 0:
                            if 'Equipo' in df.columns:
                                filtered_dfs[key] = filtered_dfs[key][
                                    filtered_dfs[key]['Equipo'].isin(selected_equipments)
                                ]
                            elif 'Equipo Estacion' in df.columns and self.analysis_type == "ADV":
                                filtered_dfs[key] = filtered_dfs[key][
                                    filtered_dfs[key]['Equipo Estacion'].isin(selected_equipments)
                                ]
                
                # Verificar si hay datos después del filtrado
                has_data = any(not df.empty for df in filtered_dfs.values())
                
                if has_data:
                    # Generar insights actualizados basados en datos filtrados
                    updated_insights = self.generate_insights(filtered_dfs)
                    
                    # Retornar todos los componentes actualizados
                    return [
                        self.create_time_trend_figure(dataframes=filtered_dfs),
                        self.create_equipment_distribution_figure(dataframes=filtered_dfs),
                        self.create_hourly_distribution_figure(dataframes=filtered_dfs, viz_type=viz_type),
                        self.create_heatmap_figure(dataframes=filtered_dfs, viz_type=viz_type),
                        [html.Li(alerta, className='alert alert-danger') for alerta in updated_insights.get('alertas_urgentes', [])]
                        if updated_insights.get('alertas_urgentes', []) else [html.P("No hay alertas urgentes en este momento", className='text-success')],
                        [html.Li(rec, className='mb-2') for rec in updated_insights.get('recomendaciones_predictivas', [])],
                        [html.Li(rec, className='mb-2') for rec in updated_insights.get('recomendaciones_preventivas', [])],
                        [html.Li(pat) for pat in updated_insights.get('patrones_detectados', [])]
                        if updated_insights.get('patrones_detectados', []) else [html.P("No se detectaron patrones significativos en los datos filtrados")]
                    ]
                
                else:
                    # Mensaje para indicar que no hay datos para el filtro seleccionado
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        title="No hay datos disponibles para los filtros seleccionados",
                        xaxis=dict(title=""),
                        yaxis=dict(title=""),
                        annotations=[dict(
                            text="Intente con un rango de fechas o equipos diferente",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False
                        )]
                    )
                    
                    # Mensaje para las recomendaciones y alertas
                    no_data_msg = [html.P("No hay datos para los filtros seleccionados", className='text-warning')]
                    
                    return empty_fig, empty_fig, empty_fig, empty_fig, no_data_msg, no_data_msg, no_data_msg, no_data_msg
                    
            except Exception as e:
                logger.error(f"Error en callback de actualización: {str(e)}")
                # Devolver componentes con mensaje de error
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Error al actualizar datos",
                    xaxis=dict(title=""),
                    yaxis=dict(title=""),
                    annotations=[dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )]
                )
                
                error_msg = [html.P(f"Error al procesar los datos: {str(e)}", className='text-danger')]
                
                return empty_fig, empty_fig, empty_fig, empty_fig, error_msg, error_msg, error_msg, error_msg
        
        # Aplicar el decorador al callback AL FINAL
        self.app.callback(
            [Output('time-trend-graph', 'figure'),
            Output('equipment-distribution', 'figure'),
            Output('hourly-distribution', 'figure'),
            Output('heatmap', 'figure'),
            Output('alertas-list', 'children'),
            Output('recomendaciones-predictivas-list', 'children'),
            Output('recomendaciones-preventivas-list', 'children'),
            Output('patrones-list', 'children')],
            [Input('apply-filters-button', 'n_clicks')],
            [State('date-range', 'start_date'),
            State('date-range', 'end_date'),
            State('equipment-filter', 'value'),
            State('visualization-type', 'value')]
        )(update_graphs_and_recommendations)
    
    def run_dashboard(self):
        """Ejecutar el dashboard web"""
        if not self.app:
            logger.error("No se ha creado el dashboard. Ejecute create_dashboard() primero.")
            return False
        
        try:
            # Configurar modo de ejecución
            server = self.app.server  # Obtener el servidor Flask subyacente
            
            # Usar threading para evitar bloqueos
            import threading
            def run_server():
                try:
                    self.app.run_server(
                        debug=False,
                        port=self.port,
                        host='localhost',
                        threaded=True,
                        use_reloader=False
                    )
                except Exception as e:
                    logger.error(f"Error en el servidor: {e}")
            
            # Iniciar servidor en un hilo separado
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            self.running = True
            logger.info(f"Dashboard de mantenimiento predictivo iniciado en puerto {self.port}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar el dashboard: {str(e)}")
            return False
        
    def stop_dashboard(self):
        """Detener el dashboard"""
        try:
            # Intentar detener el servidor
            if hasattr(self, 'app') and self.app:
                func = getattr(self.app.server, 'shutdown', None)
                if func:
                    func()
                
            self.running = False
            logger.info("Dashboard detenido")
        except Exception as e:
            logger.error(f"Error al detener el dashboard: {e}")
    
    def _run_server(self):
        """Método interno para ejecutar el servidor Dash"""
        try:
            logger.info(f"Iniciando servidor Dash en puerto {self.port}...")
            # Configuración de ejecución del servidor
            self.app.run_server(
                debug=False,
                port=self.port,
                host='localhost',
                threaded=True,
                use_reloader=False,
                dev_tools_ui=False,
                dev_tools_props_check=False
            )
        except Exception as e:
            logger.error(f"Error en el servidor Dash: {str(e)}")
            # Intentar cerrar el puerto si está en uso
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('localhost', self.port))
                s.close()
                logger.info(f"Puerto {self.port} liberado correctamente")
            except:
                logger.error(f"No se pudo liberar el puerto {self.port}")
    
    def stop_dashboard(self):
        """Detener el dashboard web"""
        self.running = False
        logger.info("Dashboard detenido")
        return True