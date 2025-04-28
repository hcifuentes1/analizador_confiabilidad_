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
from dash import dash_table, html  # Asegurarse de que dash_table esté importado


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
                
                # Imprimir rutas para verificación
                logger.info(f"Buscando archivos CDV en: {self.output_folder}")
                logger.info(f"Archivo FO: {fo_file_path} (existe: {os.path.exists(fo_file_path)})")
                
                if os.path.exists(fo_file_path):
                    self.dataframes['fallos_ocupacion'] = pd.read_csv(fo_file_path)
                    logger.info(f"Cargado archivo FO: {len(self.dataframes['fallos_ocupacion'])} filas")
                    
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
                # Cargar archivos ADV (tanto estándar como CSV)
                # Intentar diferentes variantes de nombres para mayor compatibilidad
                
                # Para discordancias
                disc_file_paths = [
                    os.path.join(self.output_folder, f'df_{self.line}_ADV_DISC_Mensual.csv'),  # Formato estándar
                    os.path.join(self.output_folder, f'df_{self.line}_ADV_DISC.csv'),          # Archivo principal
                    os.path.join(self.output_folder, f'df_{self.line}_ADV_DISC_CSV.csv')       # Formato para CSV
                ]
                
                # Para movimientos
                mov_file_paths = [
                    os.path.join(self.output_folder, f'df_{self.line}_ADV_MOV_Mensual.csv'),   # Formato estándar
                    os.path.join(self.output_folder, f'df_{self.line}_ADV_MOV.csv'),           # Archivo principal
                    os.path.join(self.output_folder, f'df_{self.line}_ADV_MOV_CSV.csv')        # Formato para CSV
                ]
                
                # Imprimir rutas para verificación
                logger.info(f"Buscando archivos ADV en: {self.output_folder}")
                
                # Cargar discordancias (usar el primer archivo que exista)
                for disc_path in disc_file_paths:
                    if os.path.exists(disc_path):
                        logger.info(f"Cargando discordancias desde: {disc_path}")
                        self.dataframes['discordancias'] = pd.read_csv(disc_path)
                        
                        # Convertir fechas
                        if 'Fecha Hora' in self.dataframes['discordancias'].columns:
                            self.dataframes['discordancias']['Fecha Hora'] = pd.to_datetime(
                                self.dataframes['discordancias']['Fecha Hora'], errors='coerce')
                        
                        logger.info(f"Cargado archivo de discordancias: {len(self.dataframes['discordancias'])} filas")
                        break
                
                # Cargar movimientos (usar el primer archivo que exista)
                for mov_path in mov_file_paths:
                    if os.path.exists(mov_path):
                        logger.info(f"Cargando movimientos desde: {mov_path}")
                        self.dataframes['movimientos'] = pd.read_csv(mov_path)
                        
                        # Convertir fechas si existen columnas de fecha
                        for date_col in ['Fecha', 'Fecha Hora']:
                            if date_col in self.dataframes['movimientos'].columns:
                                self.dataframes['movimientos'][date_col] = pd.to_datetime(
                                    self.dataframes['movimientos'][date_col], errors='coerce')
                        
                        logger.info(f"Cargado archivo de movimientos: {len(self.dataframes['movimientos'])} filas")
                        logger.info(f"Columnas en movimientos: {self.dataframes['movimientos'].columns.tolist()}")
                        break
            
            # Verificar si se cargaron datos
            if any(not df.empty for df in self.dataframes.values()):
                # Procesar columnas de anomalías en movimientos
                if 'movimientos' in self.dataframes and not self.dataframes['movimientos'].empty:
                    # Asegurar que la columna Anomalía es de tipo booleano
                    if 'Anomalía' in self.dataframes['movimientos'].columns:
                        # Convertir a booleano si es string
                        if self.dataframes['movimientos']['Anomalía'].dtype == 'object':
                            self.dataframes['movimientos']['Anomalía'] = (
                                self.dataframes['movimientos']['Anomalía']
                                .astype(str)
                                .str.lower()
                                .isin(['true', '1', 'true'])
                            )
                        # Si es numérico, convertir a booleano
                        elif self.dataframes['movimientos']['Anomalía'].dtype in ['int64', 'float64']:
                            self.dataframes['movimientos']['Anomalía'] = self.dataframes['movimientos']['Anomalía'].astype(bool)
                        
                        logger.info(f"Columna Anomalía procesada: {self.dataframes['movimientos']['Anomalía'].sum()} anomalías detectadas")
                    else:
                        # Si no existe la columna Anomalía pero tenemos Duración y umbral, crearla
                        if 'Duración (s)' in self.dataframes['movimientos'].columns and 'umbral_superior' in self.dataframes['movimientos'].columns:
                            self.dataframes['movimientos']['Anomalía'] = (
                                self.dataframes['movimientos']['Duración (s)'] > 
                                self.dataframes['movimientos']['umbral_superior']
                            )
                            logger.info(f"Columna Anomalía creada basada en umbrales: {self.dataframes['movimientos']['Anomalía'].sum()} anomalías")
                        else:
                            logger.warning("No se pudo encontrar o crear columna Anomalía en movimientos")
                
                # Generar los insights iniciales
                self.insights = self.generate_insights()
                logger.info(f"Insights generados para {self.line} - {self.analysis_type}")
                
                # Mostrar resumen de datos cargados
                for name, df in self.dataframes.items():
                    if df is not None and not df.empty:
                        logger.info(f"DataFrame '{name}': {len(df)} filas, {len(df.columns)} columnas")
                        if name == 'movimientos' and 'Anomalía' in df.columns:
                            logger.info(f"Anomalías en '{name}': {df['Anomalía'].sum()}")
                
                return True
            else:
                logger.warning("No se encontraron datos en los archivos cargados")
                return False
                
        except Exception as e:
            logger.error(f"Error al cargar los datos: {str(e)}")
            # Informar del error usando traceback para más detalle
            import traceback
            logger.error(traceback.format_exc())
            return False
        
##############################################################################################################

    def create_forecast_trend_graph(self, dataframes=None, equipment_id=None, days_ahead=30):
        """Crear gráfico con proyección de tendencia futura para un equipo"""
        # Usar dataframes filtrados si se proporcionan
        dfs = dataframes if dataframes else self.dataframes
        
        # Seleccionar datos relevantes según tipo de análisis
        if self.analysis_type == "CDV" and 'fallos_ocupacion' in dfs:
            df = dfs['fallos_ocupacion'].copy()
            date_col = 'Fecha Hora'
            equipment_col = 'Equipo'
        elif self.analysis_type == "ADV" and 'movimientos' in dfs:
            df = dfs['movimientos'].copy()
            date_col = 'Fecha'
            equipment_col = 'Equipo'
        else:
            # Si no hay datos adecuados, devolver gráfico vacío
            fig = go.Figure()
            fig.update_layout(
                title="No hay datos suficientes para generar pronósticos",
                xaxis=dict(title="Fecha"),
                yaxis=dict(title="Valor"),
                plot_bgcolor='white'
            )
            return fig
        
        # Si se especificó un equipo, filtrar para ese equipo
        if equipment_id and equipment_id in df[equipment_col].unique():
            df = df[df[equipment_col] == equipment_id]
        else:
            # Si no hay equipo específico, tomar el de mayor frecuencia
            top_equipment = df[equipment_col].value_counts().idxmax()
            df = df[df[equipment_col] == top_equipment]
            equipment_id = top_equipment
        
        # Preparar datos para análisis de tendencia
        df['date'] = pd.to_datetime(df[date_col]).dt.date
        counts_by_day = df.groupby('date').size().reset_index(name='count')
        counts_by_day['date'] = pd.to_datetime(counts_by_day['date'])
        
        # Crear serie temporal completa
        date_range = pd.date_range(
            start=counts_by_day['date'].min(), 
            end=counts_by_day['date'].max()
        )
        
        time_series = pd.DataFrame({'date': date_range})
        time_series = time_series.merge(counts_by_day, on='date', how='left')
        time_series['count'] = time_series['count'].fillna(0)
        
        # Preparar datos para el modelo
        X = np.array(range(len(time_series))).reshape(-1, 1)
        y = time_series['count'].values
        
        # Ajustar modelo de regresión
        model = LinearRegression()
        model.fit(X, y)
        
        # Calcular tendencia futura
        future_days = days_ahead
        X_future = np.array(range(len(time_series), len(time_series) + future_days)).reshape(-1, 1)
        future_dates = [time_series['date'].max() + timedelta(days=i+1) for i in range(future_days)]
        forecast = model.predict(X_future)
        forecast = np.maximum(forecast, 0)  # No permitir valores negativos
        
        # Crear figura
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=time_series['date'],
            y=time_series['count'],
            mode='lines+markers',
            name='Datos históricos',
            line=dict(color='blue')
        ))
        
        # Línea de tendencia pasada
        fig.add_trace(go.Scatter(
            x=time_series['date'],
            y=model.predict(X),
            mode='lines',
            name='Tendencia histórica',
            line=dict(color='green', dash='dash')
        ))
        
        # Proyección futura
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines',
            name='Proyección futura',
            line=dict(color='red', dash='dash')
        ))
        
        # Añadir región sombreada para la proyección
        fig.add_vrect(
            x0=time_series['date'].max(),
            x1=future_dates[-1],
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        
        # Configuración del gráfico
        fig.update_layout(
            title=f"Análisis Predictivo - {equipment_id}",
            xaxis_title="Fecha",
            yaxis_title="Frecuencia de Eventos",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text'],
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Añadir anotación con métricas predictivas
        slope = model.coef_[0]
        slope_percentage = slope / time_series['count'].mean() * 100 if time_series['count'].mean() > 0 else 0
        
        status = "Estable"
        color = "green"
        if slope_percentage > 10:
            status = "Degradación Severa"
            color = "red"
        elif slope_percentage > 5:
            status = "Degradación Moderada"
            color = "orange"
        elif slope_percentage > 2:
            status = "Degradación Leve"
            color = "blue"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Estado: <b>{status}</b><br>Tasa de cambio: {slope_percentage:.2f}% por día<br>Eventos proyectados (30 días): {round(forecast.sum())}",
            showarrow=False,
            font=dict(color=color),
            align="left",
            bgcolor="white",
            bordercolor=color,
            borderwidth=2,
            borderpad=4
        )
        
        return fig
    
    
    def create_degradation_risk_matrix(self, dataframes=None):
        """Crear matriz de riesgo de degradación para equipos"""
        # Usar dataframes filtrados si se proporcionan
        dfs = dataframes if dataframes else self.dataframes
        
        # Preparar datos según tipo de análisis
        if self.analysis_type == "CDV":
            if 'fallos_ocupacion' not in dfs or dfs['fallos_ocupacion'].empty:
                return self._create_empty_figure("No hay datos suficientes para análisis de riesgo")
            
            df = dfs['fallos_ocupacion'].copy()
            date_col = 'Fecha Hora'
            equipment_col = 'Equipo'
        
        elif self.analysis_type == "ADV":
            if 'movimientos' not in dfs or dfs['movimientos'].empty:
                return self._create_empty_figure("No hay datos suficientes para análisis de riesgo")
            
            df = dfs['movimientos'].copy()
            date_col = 'Fecha'
            equipment_col = 'Equipo'
        
        else:
            return self._create_empty_figure("Tipo de análisis no soportado para matriz de riesgo")
        
        # Convertir fechas y preparar datos
        df['date'] = pd.to_datetime(df[date_col]).dt.date
        
        # Calcular métricas por equipo
        equipment_metrics = []
        
        for equipment in df[equipment_col].unique():
            equip_df = df[df[equipment_col] == equipment]
            
            # Agrupar por fecha
            daily_counts = equip_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            if len(daily_counts) < 5:  # Necesitamos suficientes datos para análisis
                continue
                
            # Calcular tendencia
            X = np.array(range(len(daily_counts))).reshape(-1, 1)
            y = daily_counts['count'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            
            # Calcular frecuencia promedio reciente (últimos 14 días)
            recent_cutoff = datetime.now().date() - timedelta(days=14)
            recent_counts = daily_counts[daily_counts['date'] >= pd.Timestamp(recent_cutoff)]
            recent_avg = recent_counts['count'].mean() if not recent_counts.empty else 0
            
            # Calcular volatilidad (desviación estándar)
            volatility = daily_counts['count'].std()
            
            # Normalizar pendiente como porcentaje
            baseline = daily_counts['count'].mean() if daily_counts['count'].mean() > 0 else 1
            slope_pct = slope / baseline * 100
            
            # Calcular "probabilidad" (basada en frecuencia reciente)
            prob_factor = min(1.0, recent_avg / 5.0) if recent_avg > 0 else 0
            
            # Calcular "impacto" (basado en tendencia y volatilidad)
            impact_factor = min(1.0, (abs(slope_pct) / 10.0 + volatility / 3.0) / 2.0)
            
            # Calcular puntuación de riesgo (0-100)
            risk_score = 100 * prob_factor * impact_factor
            
            # Determinar categoría de riesgo
            if risk_score > 66:
                risk_category = "Alto"
                color = "red"
            elif risk_score > 33:
                risk_category = "Medio"
                color = "orange"
            else:
                risk_category = "Bajo"
                color = "green"
            
            equipment_metrics.append({
                'Equipo': equipment,
                'Frecuencia': recent_avg,
                'Tendencia': slope_pct,
                'Volatilidad': volatility,
                'Riesgo': risk_score,
                'Categoría': risk_category,
                'Color': color
            })
        
        # Si no hay métricas calculadas, mostrar figura vacía
        if not equipment_metrics:
            return self._create_empty_figure("No hay suficientes datos para análisis de riesgo")
        
        # Crear dataframe con métricas
        metrics_df = pd.DataFrame(equipment_metrics)
        
        # Ordenar por riesgo descendente
        metrics_df = metrics_df.sort_values('Riesgo', ascending=False)
        
        # Crear figura de matriz de riesgo
        fig = go.Figure()
        
        # Añadir puntos para cada equipo
        fig.add_trace(go.Scatter(
            x=metrics_df['Frecuencia'],
            y=metrics_df['Tendencia'],
            mode='markers',
            marker=dict(
                size=metrics_df['Volatilidad'] * 3 + 8,  # Tamaño basado en volatilidad
                color=metrics_df['Riesgo'],
                colorscale='RdYlGn_r',  # Rojo para alto riesgo, verde para bajo
                showscale=True,
                colorbar=dict(title="Nivel de Riesgo"),
                line=dict(width=1, color='black')
            ),
            text=metrics_df['Equipo'],
            hovertemplate='<b>%{text}</b><br>' +
                        'Frecuencia: %{x:.2f}<br>' +
                        'Tendencia: %{y:.2f}%<br>' +
                        'Volatilidad: %{marker.size:.2f}<br>' +
                        'Riesgo: %{marker.color:.1f}%<br>'
        ))
        
        # Añadir anotaciones para los equipos de mayor riesgo
        for i, row in metrics_df.head(3).iterrows():
            fig.add_annotation(
                x=row['Frecuencia'],
                y=row['Tendencia'],
                text=row['Equipo'],
                showarrow=True,
                arrowhead=1,
                arrowcolor=row['Color'],
                arrowsize=1,
                arrowwidth=2,
                ax=20,
                ay=-30,
                font=dict(color=row['Color'])
            )
        
        # Añadir zonas de riesgo como regiones rectangulares
        # Zona de alto riesgo (cuadrante superior derecho)
        fig.add_shape(
            type="rect",
            x0=metrics_df['Frecuencia'].max() * 0.6,
            y0=0,
            x1=metrics_df['Frecuencia'].max() * 1.1,
            y1=metrics_df['Tendencia'].max() * 1.1,
            line=dict(width=0),
            fillcolor="rgba(255,0,0,0.1)",
            layer="below"
        )
        
        # Añadir etiqueta para zona de alto riesgo
        fig.add_annotation(
            x=metrics_df['Frecuencia'].max() * 0.8,
            y=metrics_df['Tendencia'].max() * 0.8,
            text="ZONA DE ALTO RIESGO",
            showarrow=False,
            font=dict(color="red", size=12)
        )
        
        # Configurar layout
        fig.update_layout(
            title="Matriz de Riesgo de Degradación",
            xaxis_title="Frecuencia de Eventos (últimos 14 días)",
            yaxis_title="Tendencia de Degradación (%)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text'],
            margin=dict(l=10, r=10, t=50, b=10),
            height=500
        )
        
        return fig
        
    def _create_empty_figure(self, message):
        """Crear figura vacía con mensaje informativo"""
        fig = go.Figure()
        fig.update_layout(
            title=message,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text'],
            annotations=[
                dict(
                    text=message,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )
        return fig
    
    
    def create_degradation_pattern_analysis(self, dataframes=None):
        """Crear análisis de patrones de degradación con identificación de fases"""
        try:
            # Usar dataframes filtrados si se proporcionan
            dfs = dataframes if dataframes else self.dataframes
            
            # Preparar datos según tipo de análisis
            if self.analysis_type == "CDV":
                if 'fallos_ocupacion' not in dfs or dfs['fallos_ocupacion'].empty:
                    return self._create_empty_figure("No hay datos suficientes para análisis de patrones")
                
                df = dfs['fallos_ocupacion'].copy()
                date_col = 'Fecha Hora'
                equipment_col = 'Equipo'
            
            elif self.analysis_type == "ADV":
                if 'movimientos' not in dfs or dfs['movimientos'].empty:
                    return self._create_empty_figure("No hay datos suficientes para análisis de patrones")
                
                df = dfs['movimientos'].copy()
                # Si tenemos tiempos de duración, usarlos para análisis de degradación
                if 'Duración (s)' in df.columns:
                    date_col = 'Fecha'
                    equipment_col = 'Equipo'
                    value_col = 'Duración (s)'
                    
                    # Para el caso de ADV, usaremos el tiempo de movimiento como indicador
                    # Seleccionar el equipo con más movimientos para análisis detallado
                    top_equipment = df[equipment_col].value_counts().idxmax()
                    time_series_df = df[df[equipment_col] == top_equipment].copy()
                    
                    # Verificar que tenemos suficientes datos para continuar
                    if len(time_series_df) < 2:
                        return self._create_empty_figure(f"Datos insuficientes para el equipo {top_equipment}")
                    
                    # Ordenar por fecha
                    time_series_df['date'] = pd.to_datetime(time_series_df[date_col])
                    time_series_df = time_series_df.sort_values('date').reset_index(drop=True)  # Resetear índice después de ordenar
                    
                    # Calcular media móvil de tiempos de movimiento
                    time_series_df['moving_avg'] = time_series_df[value_col].rolling(
                        window=min(5, len(time_series_df)),  # Usar window más pequeña si no hay suficientes datos
                        min_periods=1
                    ).mean()
                    
                    # Calcular desviación estándar móvil
                    time_series_df['moving_std'] = time_series_df[value_col].rolling(
                        window=min(5, len(time_series_df)),  # Usar window más pequeña si no hay suficientes datos
                        min_periods=1
                    ).std()
                    
                    # Identificar fases de degradación
                    global_mean = time_series_df[value_col].mean()
                    global_std = max(time_series_df[value_col].std(), 0.001)  # Evitar división por cero o valores muy pequeños
                    
                    time_series_df['phase'] = 1  # Fase normal por defecto
                    time_series_df.loc[time_series_df[value_col] > global_mean + global_std, 'phase'] = 2
                    time_series_df.loc[time_series_df[value_col] > global_mean + 2*global_std, 'phase'] = 3
                    
                    # Crear figura
                    fig = go.Figure()
                    
                    # Añadir línea de valores reales
                    fig.add_trace(go.Scatter(
                        x=time_series_df['date'],
                        y=time_series_df[value_col],
                        mode='markers+lines',
                        name='Tiempo de movimiento',
                        marker=dict(
                            color=time_series_df['phase'].map({
                                1: 'green',
                                2: 'orange',
                                3: 'red'
                            })
                        ),
                        line=dict(color='blue', width=1)
                    ))
                    
                    # Añadir media móvil
                    fig.add_trace(go.Scatter(
                        x=time_series_df['date'],
                        y=time_series_df['moving_avg'],
                        mode='lines',
                        name='Media móvil (5 puntos)',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # Añadir líneas de umbral
                    fig.add_trace(go.Scatter(
                        x=time_series_df['date'],
                        y=[global_mean] * len(time_series_df),
                        mode='lines',
                        name='Media global',
                        line=dict(color='black', width=1, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=time_series_df['date'],
                        y=[global_mean + global_std] * len(time_series_df),
                        mode='lines',
                        name='Umbral de precaución (+1σ)',
                        line=dict(color='orange', width=1, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=time_series_df['date'],
                        y=[global_mean + 2*global_std] * len(time_series_df),
                        mode='lines',
                        name='Umbral de alerta (+2σ)',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    
                    # Encontrar puntos de cambio de fase para destacarlos
                    phase_changes = []
                    
                    if not time_series_df.empty:
                        current_phase = time_series_df['phase'].iloc[0]
                        
                        for i, row in time_series_df.iterrows():
                            if row['phase'] != current_phase:
                                phase_changes.append({
                                    'date': row['date'],
                                    'value': row[value_col],
                                    'old_phase': current_phase,
                                    'new_phase': row['phase']
                                })
                                current_phase = row['phase']
                    
                    # Añadir anotaciones para los cambios de fase
                    for change in phase_changes:
                        direction = "↑" if change['new_phase'] > change['old_phase'] else "↓"
                        color = "red" if change['new_phase'] > change['old_phase'] else "green"
                        
                        fig.add_annotation(
                            x=change['date'],
                            y=change['value'],
                            text=f"Fase {change['old_phase']}{direction}Fase {change['new_phase']}",
                            showarrow=True,
                            arrowhead=1,
                            arrowcolor=color,
                            arrowsize=1,
                            arrowwidth=2,
                            ax=0,
                            ay=-40
                        )
                    
                    # CORRECCIÓN: Manejar las secciones sombreadas para cada fase con índices seguros
                    # Identificar rangos continuos por fase
                    def get_continuous_ranges(phase_series, phase_value):
                        """Obtener rangos continuos de una fase específica"""
                        ranges = []
                        start_idx = None
                        
                        for idx, value in enumerate(phase_series):
                            if value == phase_value and start_idx is None:
                                start_idx = idx
                            elif value != phase_value and start_idx is not None:
                                ranges.append((start_idx, idx - 1))
                                start_idx = None
                        
                        # No olvidar el último rango si termina con la fase indicada
                        if start_idx is not None:
                            ranges.append((start_idx, len(phase_series) - 1))
                        
                        return ranges
                    
                    # Procesamiento de cada fase con manejo seguro de índices
                    for phase_value, color in [(1, "rgba(0,255,0,0.1)"), 
                                            (2, "rgba(255,165,0,0.1)"), 
                                            (3, "rgba(255,0,0,0.1)")]:
                        
                        phase_ranges = get_continuous_ranges(time_series_df['phase'], phase_value)
                        
                        for start_idx, end_idx in phase_ranges:
                            # Verificar que los índices son válidos
                            if 0 <= start_idx < len(time_series_df) and 0 <= end_idx < len(time_series_df):
                                fig.add_shape(
                                    type="rect",
                                    x0=time_series_df.iloc[start_idx]['date'],
                                    x1=time_series_df.iloc[end_idx]['date'],
                                    y0=time_series_df[value_col].min() * 0.9,
                                    y1=time_series_df[value_col].max() * 1.1,
                                    fillcolor=color,
                                    line=dict(width=0),
                                    layer="below"
                                )
                    
                    # Calcular características de degradación
                    if len(time_series_df) >= 10:
                        # Ajustar modelo de regresión para determinar tendencia
                        X = np.array(range(len(time_series_df))).reshape(-1, 1)
                        y = time_series_df[value_col].values
                        
                        try:
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            slope = model.coef_[0]
                            slope_pct = slope / time_series_df[value_col].mean() * 100
                            
                            # Determinar fase actual del sistema
                            current_phase = time_series_df['phase'].iloc[-1]
                            phase_names = {1: "Normal", 2: "Precaución", 3: "Alerta"}
                            
                            # Predecir momento de fallo
                            if slope > 0:  # Tendencia creciente = degradación
                                time_to_critical = (global_mean + 3*global_std - time_series_df[value_col].iloc[-1]) / slope
                                time_to_critical = max(0, time_to_critical)  # No permitir valores negativos
                                
                                failure_date = time_series_df['date'].iloc[-1] + pd.Timedelta(days=time_to_critical)
                                
                                # Añadir anotación con predicción
                                fig.add_annotation(
                                    x=0.02,
                                    y=0.02,
                                    xref="paper",
                                    yref="paper",
                                    text=f"<b>Análisis de Degradación</b><br>" +
                                        f"Fase actual: <b>{phase_names[current_phase]}</b><br>" +
                                        f"Tasa de degradación: {slope_pct:.2f}% por día<br>" +
                                        f"Tiempo estimado hasta fallo: {time_to_critical:.1f} días<br>" +
                                        f"Fecha estimada de fallo: {failure_date.strftime('%d/%m/%Y')}",
                                    showarrow=False,
                                    align="left",
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1,
                                    borderpad=4
                                )
                            else:
                                # No hay degradación (tendencia estable o decreciente)
                                fig.add_annotation(
                                    x=0.02,
                                    y=0.02,
                                    xref="paper",
                                    yref="paper",
                                    text=f"<b>Análisis de Degradación</b><br>" +
                                        f"Fase actual: <b>{phase_names[current_phase]}</b><br>" +
                                        f"No se detecta degradación progresiva<br>" +
                                        f"Tendencia: {slope_pct:.2f}% por día",
                                    showarrow=False,
                                    align="left",
                                    bgcolor="white",
                                    bordercolor="black",
                                    borderwidth=1,
                                    borderpad=4
                                )
                        except Exception as e:
                            logger.warning(f"Error al calcular regresión: {str(e)}")
                    
                    # Configurar layout
                    fig.update_layout(
                        title=f"Análisis de Patrones de Degradación - {top_equipment}",
                        xaxis_title="Fecha",
                        yaxis_title="Tiempo de Movimiento (s)",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color=self.colors['text'] if hasattr(self, 'colors') and 'text' in self.colors else 'black',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=10, r=10, t=50, b=10),
                        height=500
                    )
                    
                    return fig
                
                else:
                    return self._create_empty_figure("No hay datos de tiempo de movimiento para análisis de degradación")
            
            else:
                return self._create_empty_figure("Tipo de análisis no soportado para patrones de degradación")
        
        except Exception as e:
            logger.error(f"Error en create_degradation_pattern_analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_empty_figure(f"Error en análisis de patrones: {str(e)}")


##################################################################################################################
    
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
        
    def create_timeline_figure(self, dataframes=None):
        """Crear gráfico de línea de tiempo de movimientos por hora del día"""
        # Usar los dataframes filtrados si se proporcionan, o los originales si no
        dfs = dataframes if dataframes else self.dataframes
        
        if self.analysis_type == "ADV" and 'movimientos' in dfs and not dfs['movimientos'].empty:
            df = dfs['movimientos'].copy()
            
            # Extraer hora del día para el eje X
            if 'Hora Inicio' in df.columns:
                # Convertir a datetime si es string
                if isinstance(df['Hora Inicio'].iloc[0], str):
                    try:
                        # Primero intentar con formato estándar
                        df['hora_numeric'] = pd.to_datetime(df['Hora Inicio'], format='%H:%M:%S').dt.hour + \
                                            pd.to_datetime(df['Hora Inicio'], format='%H:%M:%S').dt.minute / 60
                    except ValueError:
                        # Si falla, usar un enfoque más flexible que maneje milisegundos
                        df['hora_numeric'] = df['Hora Inicio'].apply(
                            lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if ':' in x else float(x)
                        )
                else:
                    # Si ya es tipo time
                    df['hora_numeric'] = df['Hora Inicio'].apply(lambda x: x.hour + x.minute / 60)
            
            # Crear figura
            fig = go.Figure()
            
            # Añadir línea de tiempo y duración
            fig.add_trace(go.Scatter(
                x=df['hora_numeric'],
                y=df['Duración (s)'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['Duración (s)'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Duración (s)")
                ),
                text=df['Equipo'],
                hovertemplate='<b>Equipo:</b> %{text}<br>' +
                            '<b>Hora:</b> %{x:.2f}<br>' +
                            '<b>Duración:</b> %{y:.2f} s<br>'
            ))
            
            # Añadir línea de umbral si existe
            if 'umbral_superior' in df.columns:
                # Usar el umbral promedio para simplificar
                umbral_medio = df['umbral_superior'].mean()
                fig.add_shape(
                    type="line",
                    x0=df['hora_numeric'].min(),
                    y0=umbral_medio,
                    x1=df['hora_numeric'].max(),
                    y1=umbral_medio,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    ),
                    name="Umbral"
                )
                
                # Añadir anotación explicando el umbral
                fig.add_annotation(
                    x=df['hora_numeric'].max(),
                    y=umbral_medio,
                    text=f"Umbral: {umbral_medio:.2f}s",
                    showarrow=True,
                    arrowhead=1,
                    ax=50,
                    ay=0
                )
            
            # Configurar layout
            fig.update_layout(
                title="Tiempos de Movimiento por Hora del Día",
                xaxis_title="Hora del día",
                yaxis_title="Duración del movimiento (s)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_color=self.colors['text'],
                margin=dict(l=10, r=10, t=50, b=10),
                height=400
            )
            
            # Mejorar formato del eje X (horas)
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(0, 25, 2)),
                ticktext=[f"{h}:00" for h in range(0, 25, 2)]
            )
            
            return fig
        
        # Figura vacía en caso de no tener datos
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos disponibles para mostrar línea de tiempo",
            xaxis=dict(title="Hora del día"),
            yaxis=dict(title="Duración (s)"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text']
        )
        
        return fig

    
    def create_dashboard(self):
        """Crear y configurar el dashboard web"""
        if not self.dataframes:
            logger.error("No hay datos cargados para generar el dashboard")
            return False
        
        try:
            # Generar insights iniciales si no existen
            if not hasattr(self, 'insights') or not self.insights:
                self.insights = self.generate_insights()
                
            # Después de cargar los dataframes pero antes de crear el dashboard
            if hasattr(self, 'dataframes') and self.dataframes:
                logger.info(f"Dashboard: Dataframes cargados: {list(self.dataframes.keys())}")
                
                # Depuración especial para ADV
                if hasattr(self, 'analysis_type') and self.analysis_type == "ADV":
                    if 'movimientos' in self.dataframes:
                        logger.info(f"Dashboard: Datos de movimientos: {len(self.dataframes['movimientos'])} filas")
                        if 'Equipo' in self.dataframes['movimientos'].columns:
                            equipos = self.dataframes['movimientos']['Equipo'].unique()
                            logger.info(f"Dashboard: Equipos disponibles: {equipos}")
                        else:
                            logger.info(f"Dashboard: Columnas en movimientos: {self.dataframes['movimientos'].columns.tolist()}")
            
            # Verificar que los atributos necesarios existen
            if not hasattr(self, 'line') or not hasattr(self, 'analysis_type'):
                logger.error("Atributos 'line' o 'analysis_type' no definidos")
                return False
                
            # Verificar que se han definido los colores
            if not hasattr(self, 'colors'):
                self.colors = {
                    'primary': '#3498db',
                    'secondary': '#2C3E50',
                    'success': '#2ecc71',
                    'danger': '#e74c3c',
                    'warning': '#f39c12',
                    'card_background': '#ffffff'
                }
            
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
                    html.Div(className='row mb-4 g-3', children=self.create_kpi_cards()),
                    
                    # Fila para gráfico de línea de tiempo (nueva fila)
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Línea de Tiempo de Movimientos", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='timeline-graph', figure=self.create_timeline_figure())
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila para gráfico de proyección de tendencia
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Proyección de Tendencia", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='forecast-trend-graph', figure=self.create_forecast_trend_graph())
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila para matriz de riesgo de degradación
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Matriz de Riesgo de Degradación", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='degradation-risk-matrix', figure=self.create_degradation_risk_matrix())
                                ])
                            ])
                        ])
                    ]),
                    
                    # Fila para análisis de patrones de degradación
                    html.Div(className='row mb-4', children=[
                        html.Div(className='col-md-12', children=[
                            html.Div(className='card', style={'backgroundColor': self.colors['card_background']}, children=[
                                html.Div(className='card-header', children=[
                                    html.H5("Análisis de Patrones de Degradación", className='card-title')
                                ]),
                                html.Div(className='card-body', children=[
                                    dcc.Graph(id='degradation-pattern-analysis', figure=self.create_degradation_pattern_analysis())
                                ])
                            ])
                        ])
                    ]),
                    
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
                    
                    # Fila de alertas urgentes
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
                                        for rec in self.insights.get('recomendaciones_predictivas', []) or []
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
                                        for rec in self.insights.get('recomendaciones_preventivas', []) or []
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
                                        html.Li(pat) for pat in self.insights.get('patrones_detectados', []) or []
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
            import traceback
            logger.error(traceback.format_exc())
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
                    
            elif self.analysis_type == "ADV":
                # KPIs para ADV
                
                # Total de discordancias
                total_disc = len(self.dataframes.get('discordancias', pd.DataFrame()))
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#E74C3C', 'borderBottom': 'none'}, 
                                    children=[html.H5("Discordancias", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{total_disc}", className='display-4 mb-0 fw-bold'),
                                html.P("Total detectadas", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Total de movimientos anómalos
                mov_anomalos = 0
                if 'movimientos' in self.dataframes and 'Anomalía' in self.dataframes['movimientos'].columns:
                    mov_anomalos = self.dataframes['movimientos']['Anomalía'].sum()
                    
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#F39C12', 'borderBottom': 'none'}, 
                                    children=[html.H5("Movimientos Anómalos", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{mov_anomalos}", className='display-4 mb-0 fw-bold'),
                                html.P("Tiempo excesivo", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Equipos con tendencia degradada
                equipos_tendencia = 0
                if hasattr(self, 'df_L2_ADV_TIME') and 'trend_significant' in self.df_L2_ADV_TIME.columns:
                    equipos_tendencia = self.df_L2_ADV_TIME['trend_significant'].sum()
                
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#9B59B6', 'borderBottom': 'none'}, 
                                    children=[html.H5("Degradación Progresiva", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{equipos_tendencia}", className='display-4 mb-0 fw-bold'),
                                html.P("Agujas con tendencia creciente", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Equipos en estado crítico
                equipos_criticos = 0
                if hasattr(self, 'df_L2_ADV_TIME') and 'alerta_nivel' in self.df_L2_ADV_TIME.columns:
                    equipos_criticos = (self.df_L2_ADV_TIME['alerta_nivel'] == 'Crítico').sum()
                else:
                    # Alternativa si no está disponible en el dataframe de tiempos
                    if hasattr(self, 'insights') and self.insights and 'alertas_urgentes' in self.insights:
                        equipos_criticos = len(self.insights['alertas_urgentes'])
                
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#C0392B', 'borderBottom': 'none'}, 
                                    children=[html.H5("Agujas Críticas", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{equipos_criticos}", className='display-4 mb-0 fw-bold'),
                                html.P("Requieren atención inmediata", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Tiempo promedio de movimiento
                tiempo_promedio = "N/A"
                if 'movimientos' in self.dataframes and 'Duración (s)' in self.dataframes['movimientos'].columns:
                    tiempo_promedio = f"{self.dataframes['movimientos']['Duración (s)'].mean():.2f}s"
                    
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': '#3498DB', 'borderBottom': 'none'}, 
                                    children=[html.H5("Tiempo Promedio", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.H1(f"{tiempo_promedio}", className='display-4 mb-0 fw-bold'),
                                html.P("Duración media de movimiento", className='card-text text-muted')
                            ])
                        ])
                    ])
                )
                
                # Estado general del sistema
                estado_sistema = "NORMAL"
                estado_color = "#2ECC71"  # Verde por defecto
                
                if equipos_tendencia > 0:
                    estado_sistema = "ATENCIÓN PREVENTIVA"
                    estado_color = "#3498DB"  # Azul
                    
                if equipos_criticos > 0:
                    estado_sistema = "REQUIERE ATENCIÓN"
                    estado_color = "#F39C12"  # Amarillo
                    
                if equipos_criticos > 2 or total_disc > 5:
                    estado_sistema = "CRÍTICO"
                    estado_color = "#E74C3C"  # Rojo
                
                kpi_cards.append(
                    html.Div(className='col-md-4 col-lg-3 mb-3', children=[
                        html.Div(className='card h-100 border-0 shadow-sm', style={'borderRadius': '10px', 'overflow': 'hidden'}, children=[
                            html.Div(className='card-header text-white text-center py-3', 
                                    style={'backgroundColor': estado_color, 'borderBottom': 'none'}, 
                                    children=[html.H5("Estado del Sistema", className='m-0 fw-bold')]),
                            html.Div(className='card-body text-center d-flex flex-column justify-content-center', children=[
                                html.Div([
                                    html.Span("✅" if estado_sistema == "NORMAL" else 
                                            "ℹ️" if estado_sistema == "ATENCIÓN PREVENTIVA" else
                                            "⚠️" if estado_sistema == "REQUIERE ATENCIÓN" else "❌", 
                                            style={'fontSize': '2rem', 'marginRight': '10px'}),
                                    html.H2(f"{estado_sistema}", 
                                        className='mb-0 fw-bold d-inline',
                                        style={'color': estado_color})
                                ]),
                                html.P("Evaluación general", className='card-text text-muted')
                            ])
                        ])
                    ])
                )

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
            if 'movimientos' in dfs and not dfs['movimientos'].empty:
                df = dfs['movimientos'].copy()
                
                # Impresión de diagnóstico
                print(f"create_time_trend_figure: Columnas en df: {df.columns.tolist()}")
                print(f"create_time_trend_figure: Primeras filas de df: {df.head(2).to_dict('records')}")
                
                # Convertir 'Fecha' a datetime si es string
                if 'Fecha' in df.columns and isinstance(df['Fecha'].iloc[0], str):
                    df['Fecha'] = pd.to_datetime(df['Fecha'])
                
                # Agrupar por fecha para contar movimientos
                if 'Fecha' in df.columns:
                    # Asegurarse de que Fecha sea fecha (no string)
                    if df['Fecha'].dtype != 'datetime64[ns]':
                        df['fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
                    else:
                        df['fecha_dt'] = df['Fecha']
                    
                    # Agrupar por fecha
                    mov_por_dia = df.groupby('fecha_dt').size().reset_index(name='conteo')
                    
                    # Ordenar por fecha
                    mov_por_dia = mov_por_dia.sort_values('fecha_dt')
                    
                    # Crear figura
                    fig = px.line(
                        mov_por_dia, 
                        x='fecha_dt', 
                        y='conteo',
                        labels={'fecha_dt': 'Fecha', 'conteo': 'Número de Movimientos'},
                        title="Tendencia de Movimientos de Agujas",
                        color_discrete_sequence=[self.colors['primary']]
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color=self.colors['text'],
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    
                    return fig
            
            # Si no hay datos de movimientos, verificar discordancias
            if 'discordancias' in dfs and not dfs['discordancias'].empty:
                # Código para mostrar tendencia de discordancias...
                pass
        
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
            if 'movimientos' in dfs and not dfs['movimientos'].empty:
                df = dfs['movimientos'].copy()
                
                # Impresión de diagnóstico
                print(f"create_equipment_distribution_figure: Columnas en df: {df.columns.tolist()}")
                
                # Contar movimientos por equipo
                if 'Equipo' in df.columns:
                    mov_por_equipo = df['Equipo'].value_counts().reset_index()
                    mov_por_equipo.columns = ['Equipo', 'Conteo']
                    
                    # Ordenar por conteo descendente
                    mov_por_equipo = mov_por_equipo.sort_values('Conteo', ascending=False)
                    
                    # Crear figura
                    fig = px.bar(
                        mov_por_equipo, 
                        x='Equipo', 
                        y='Conteo',
                        labels={'Equipo': 'Aparato de Vía', 'Conteo': 'Número de Movimientos'},
                        title="Distribución de Movimientos por Aparato de Vía",
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
            
            # Si no hay datos de movimientos, verificar discordancias
            if 'discordancias' in dfs and not dfs['discordancias'].empty:
                # Código para mostrar distribución de discordancias...
                pass
        
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
    
    def extract_station_code(self, equipo):
        """Extrae correctamente el código de estación de un equipo"""
        if not equipo or pd.isna(equipo):
            return "NA"
            
        equipo_str = str(equipo)
        
        # Caso 1: El formato es "Kag XX/YY ZZ" donde ZZ es la estación (2 letras al final)
        if "Kag" in equipo_str and "/" in equipo_str and len(equipo_str) > 3:
            parts = equipo_str.split()
            if len(parts) > 1:
                last_part = parts[-1]
                # Si la última parte tiene 2 letras y son mayúsculas, probablemente es la estación
                if len(last_part) == 2 and last_part.isalpha() and last_part.isupper():
                    return last_part
        
        # Caso 2: Formato con espacios "Kag XX ZZ" donde ZZ es la estación
        parts = equipo_str.split()
        if len(parts) > 1:
            last_part = parts[-1]
            # Si la última parte tiene 2 letras y son mayúsculas, probablemente es la estación
            if len(last_part) == 2 and last_part.isalpha() and last_part.isupper():
                return last_part
        
        # Caso 3: Extraer dos letras mayúsculas al final del string
        import re
        match = re.search(r'[A-Z]{2}$', equipo_str)
        if match:
            return match.group(0)
            
        # Fallback para formatos desconocidos
        return "AV"  # Valor por defecto
    
    # 3. Mejora en la creación de la tabla de datos
    def create_data_table(self):
        """Crear tabla de datos con detalle de motivos y resaltado de anomalías"""
        try:
            if self.analysis_type == "CDV":
                if 'fallos_ocupacion' in self.dataframes and not self.dataframes['fallos_ocupacion'].empty:
                    df = self.dataframes['fallos_ocupacion'].copy()
                    
                    # Seleccionar columnas relevantes
                    if 'Fecha Hora' in df.columns and 'Equipo' in df.columns and 'Estacion' in df.columns:
                        # Asegurar que Fecha Hora es datetime
                        df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora'], errors='coerce')
                        
                        # Seleccionar columnas más relevantes
                        if 'Diff.Time_+1_row' in df.columns:
                            df = df[['Fecha Hora', 'Equipo', 'Estacion', 'Diff.Time_+1_row']]
                            df.rename(columns={'Diff.Time_+1_row': 'Tiempo (s)'}, inplace=True)
                        else:
                            df = df[['Fecha Hora', 'Equipo', 'Estacion']]
                        
                        # Formatear para mostrar
                        df['Fecha Hora'] = df['Fecha Hora'].dt.strftime('%d-%m-%Y %H:%M:%S')
                        
                        # Limitar a 50 filas para mejor rendimiento
                        df = df.head(50)
                        
                        # Usar dash_table.DataTable para mejor rendimiento
                        return dash_table.DataTable(
                            id='data-table',
                            columns=[{"name": col, "id": col} for col in df.columns],
                            data=df.to_dict('records'),
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '5px',
                                'backgroundColor': 'white'
                            },
                            style_header={
                                'backgroundColor': '#f8f9fa',
                                'fontWeight': 'bold'
                            },
                            page_size=15,
                            page_action='native',
                            sort_action='native',
                            filter_action='native'
                        )
            
            elif self.analysis_type == "ADV":
                # Primero intentamos mostrar movimientos anómalos si existen
                if 'movimientos' in self.dataframes and not self.dataframes['movimientos'].empty:
                    df = self.dataframes['movimientos'].copy()
                    
                    # Verificar si existe la columna Anomalía
                    has_anomaly_column = 'Anomalía' in df.columns
                    
                    # Añadir columna de motivo
                    df['Motivo'] = "Movimiento normal"
                    
                    # Marcar movimientos anómalos si la columna existe
                    if has_anomaly_column:
                        # Convertir a booleano si es necesario
                        if df['Anomalía'].dtype == 'object':
                            df['Anomalía'] = df['Anomalía'].astype(str).str.lower() == 'true'
                        
                        # Marcar movimientos anómalos
                        df.loc[df['Anomalía'] == True, 'Motivo'] = "Tiempo de movimiento anómalo"
                        
                    # Asegurar que Estación está incluida
                    if 'Estación' not in df.columns and 'Equipo' in df.columns:
                        df['Estación'] = df['Equipo'].apply(self.extract_station_code)
                    
                    # Seleccionar columnas relevantes para mostrar
                    columns_to_show = ['Fecha', 'Hora Inicio', 'Hora Fin', 'Duración (s)', 'Equipo', 'Motivo', 'Estación', 'Linea']
                    columns_present = [col for col in columns_to_show if col in df.columns]
                    
                    if columns_present:
                        df_final = df[columns_present]
                        
                        # Ordenar por motivo y duración si están disponibles
                        if 'Motivo' in df_final.columns and 'Duración (s)' in df_final.columns:
                            df_final['es_anormal'] = df_final['Motivo'] != "Movimiento normal"
                            df_final = df_final.sort_values(['es_anormal', 'Duración (s)'], ascending=[False, False])
                            if 'es_anormal' in df_final.columns:
                                df_final = df_final.drop(columns=['es_anormal'])
                        
                        # Crear tabla con paginación
                        return dash_table.DataTable(
                            id='data-table',
                            columns=[{"name": col, "id": col} for col in df_final.columns],
                            data=df_final.to_dict('records'),
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '5px',
                                'backgroundColor': 'white'
                            },
                            style_header={
                                'backgroundColor': '#f8f9fa',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Motivo} contains "anómalo"'},
                                    'backgroundColor': 'rgba(255,200,200,0.3)',
                                    'color': 'red',
                                    'fontWeight': 'bold'
                                }
                            ],
                            page_size=15,
                            page_action='native',
                            sort_action='native',
                            filter_action='native'
                        )
            
            # Si llegamos aquí es porque no pudimos crear la tabla
            return html.Div("No hay datos disponibles para mostrar en la tabla.", className="alert alert-warning")
        
        except Exception as e:
            import traceback
            logger.error(f"Error al crear tabla de datos: {str(e)}")
            logger.error(traceback.format_exc())
            return html.Div(f"Error al mostrar datos: {str(e)}", className="alert alert-danger")

    
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
            # Primero intentar con dataframes específicos
            for df_name in ['discordancias', 'movimientos']:
                if df_name in self.dataframes and 'Equipo' in self.dataframes[df_name].columns:
                    equipos.extend(self.dataframes[df_name]['Equipo'].unique())
            
            # Como fallback, intentar buscar en todos los dataframes
            if not equipos:
                for df_name, df in self.dataframes.items():
                    if 'Equipo' in df.columns:
                        equipos.extend(df['Equipo'].unique())
                        print(f"Dashboard: Equipos encontrados en {df_name}: {df['Equipo'].unique()}")
        
        return sorted(list(set(equipos)))
    
    def apply_filters(self, start_date, end_date, selected_equipments):
        """Aplicar filtros a todos los dataframes"""
        filtered_dfs = {}
        
        # Convertir fechas a datetime
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
        
        logger.info(f"Aplicando filtros a dataframes: {list(self.dataframes.keys())}")
        
        for key, df in self.dataframes.items():
            if df is not None and not df.empty:
                filtered_dfs[key] = df.copy()
                before_rows = len(filtered_dfs[key])
                
                # Filtrar por fecha
                if start_date and end_date:
                    if 'Fecha Hora' in df.columns:
                        # Asegurar que Fecha Hora es datetime
                        filtered_dfs[key]['Fecha Hora'] = pd.to_datetime(filtered_dfs[key]['Fecha Hora'], errors='coerce')
                        filtered_dfs[key] = filtered_dfs[key][
                            (filtered_dfs[key]['Fecha Hora'] >= start_date) & 
                            (filtered_dfs[key]['Fecha Hora'] <= end_date)
                        ]
                    elif 'Fecha' in df.columns:
                        # Asegurar que Fecha es datetime
                        filtered_dfs[key]['Fecha'] = pd.to_datetime(filtered_dfs[key]['Fecha'], errors='coerce')
                        filtered_dfs[key] = filtered_dfs[key][
                            (filtered_dfs[key]['Fecha'] >= start_date) & 
                            (filtered_dfs[key]['Fecha'] <= end_date)
                        ]
                
                after_date_filter = len(filtered_dfs[key])
                logger.info(f"DataFrame {key}: {before_rows} filas → {after_date_filter} filas después de filtrar por fecha")
                
                # Filtrar por equipamiento
                if selected_equipments and len(selected_equipments) > 0:
                    before_equip = len(filtered_dfs[key])
                    
                    if 'Equipo' in df.columns:
                        filtered_dfs[key] = filtered_dfs[key][
                            filtered_dfs[key]['Equipo'].isin(selected_equipments)
                        ]
                    elif 'Equipo Estacion' in df.columns:
                        filtered_dfs[key] = filtered_dfs[key][
                            filtered_dfs[key]['Equipo Estacion'].isin(selected_equipments)
                        ]
                    
                    after_equip = len(filtered_dfs[key])
                    logger.info(f"DataFrame {key}: {before_equip} filas → {after_equip} filas después de filtrar por equipos")
        
        return filtered_dfs
    
# 4. Actualizar los callbacks para usar correctamente el método apply_filters
    def setup_callbacks(self):
        """Configurar callbacks para interactividad"""
        if not self.app:
            return
        
        # Callback para el botón de filtros
        @self.app.callback(
            [Output('time-trend-graph', 'figure'),
            Output('equipment-distribution', 'figure'),
            Output('hourly-distribution', 'figure'),
            Output('heatmap', 'figure'),
            Output('timeline-graph', 'figure')],
            [Input('apply-filters-button', 'n_clicks')],
            [State('date-range', 'start_date'),
            State('date-range', 'end_date'),
            State('equipment-filter', 'value'),
            State('visualization-type', 'value')]
        )
        def update_graphs(n_clicks, start_date, end_date, selected_equipments, viz_type):
            """Callback para actualizar los gráficos cuando se aplican filtros"""
            logger.info(f"Callback de filtros activado. n_clicks: {n_clicks}")
            logger.info(f"Fechas: {start_date} a {end_date}")
            logger.info(f"Equipos seleccionados: {selected_equipments}")
            logger.info(f"Tipo visualización: {viz_type}")
            
            # Si es la primera carga (no se ha hecho click) o no hay filtros seleccionados
            if n_clicks is None:
                return [
                    self.create_time_trend_figure(),
                    self.create_equipment_distribution_figure(),
                    self.create_hourly_distribution_figure(),
                    self.create_heatmap_figure(),
                    self.create_timeline_figure()
                ]
            
            # Aplicar filtros a los dataframes
            try:
                filtered_dfs = self.apply_filters(start_date, end_date, selected_equipments)
                
                # Retornar gráficos actualizados
                return [
                    self.create_time_trend_figure(dataframes=filtered_dfs),
                    self.create_equipment_distribution_figure(dataframes=filtered_dfs),
                    self.create_hourly_distribution_figure(dataframes=filtered_dfs, viz_type=viz_type),
                    self.create_heatmap_figure(dataframes=filtered_dfs, viz_type=viz_type),
                    self.create_timeline_figure(dataframes=filtered_dfs)
                ]
            except Exception as e:
                logger.error(f"Error en callback update_graphs: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # En caso de error, devolver los gráficos sin filtrar
                return [
                    self.create_time_trend_figure(),
                    self.create_equipment_distribution_figure(),
                    self.create_hourly_distribution_figure(),
                    self.create_heatmap_figure(),
                    self.create_timeline_figure()
                ]
        
        # Callback separado para recomendaciones y patrones
        @self.app.callback(
            [Output('alertas-list', 'children'),
            Output('recomendaciones-predictivas-list', 'children'),
            Output('recomendaciones-preventivas-list', 'children'),
            Output('patrones-list', 'children')],
            [Input('apply-filters-button', 'n_clicks')],
            [State('date-range', 'start_date'),
            State('date-range', 'end_date'),
            State('equipment-filter', 'value')]
        )
        def update_recommendations(n_clicks, start_date, end_date, selected_equipments):
            """Callback para actualizar recomendaciones cuando se aplican filtros"""
            # Si es la primera carga
            if n_clicks is None:
                return [
                    [html.Li(alerta, className='alert alert-danger') for alerta in self.insights.get('alertas_urgentes', [])] 
                    if self.insights.get('alertas_urgentes', []) else [html.P("No hay alertas urgentes en este momento", className='text-success')],
                    [html.Li(rec, className='mb-2') for rec in self.insights.get('recomendaciones_predictivas', [])],
                    [html.Li(rec, className='mb-2') for rec in self.insights.get('recomendaciones_preventivas', [])],
                    [html.Li(pat) for pat in self.insights.get('patrones_detectados', [])]
                    if self.insights.get('patrones_detectados', []) else [html.P("No se detectaron patrones significativos")]
                ]
            
            try:
                # Aplicar filtros a los dataframes
                filtered_dfs = self.apply_filters(start_date, end_date, selected_equipments)
                
                # Generar insights basados en datos filtrados
                updated_insights = self.generate_insights(filtered_dfs)
                
                # Retornar recomendaciones actualizadas
                return [
                    [html.Li(alerta, className='alert alert-danger') for alerta in updated_insights.get('alertas_urgentes', [])]
                    if updated_insights.get('alertas_urgentes', []) else [html.P("No hay alertas urgentes en este momento", className='text-success')],
                    [html.Li(rec, className='mb-2') for rec in updated_insights.get('recomendaciones_predictivas', [])],
                    [html.Li(rec, className='mb-2') for rec in updated_insights.get('recomendaciones_preventivas', [])],
                    [html.Li(pat) for pat in updated_insights.get('patrones_detectados', [])]
                    if updated_insights.get('patrones_detectados', []) else [html.P("No se detectaron patrones significativos en los datos filtrados")]
                ]
            except Exception as e:
                logger.error(f"Error en callback update_recommendations: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # En caso de error, devolver las recomendaciones originales
                return [
                    [html.Li(alerta, className='alert alert-danger') for alerta in self.insights.get('alertas_urgentes', [])] 
                    if self.insights.get('alertas_urgentes', []) else [html.P("No hay alertas urgentes en este momento", className='text-success')],
                    [html.Li(rec, className='mb-2') for rec in self.insights.get('recomendaciones_predictivas', [])],
                    [html.Li(rec, className='mb-2') for rec in self.insights.get('recomendaciones_preventivas', [])],
                    [html.Li(pat) for pat in self.insights.get('patrones_detectados', [])]
                    if self.insights.get('patrones_detectados', []) else [html.P("No se detectaron patrones significativos")]
                ]
    
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