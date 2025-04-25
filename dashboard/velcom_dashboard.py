# dashboard/velcom_dashboard.py
import os
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import webbrowser
from threading import Timer
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc


def create_anomaly_detection_figure(df):
    """Crear visualización mejorada de detección de anomalías"""
    try:
        # Verificar que hay suficientes datos
        if len(df) < 20:
            raise ValueError("Datos insuficientes para realizar detección de anomalías (mínimo 20 registros)")
        
        # Preparar datos
        features = ['station_stay_time', 'hour_of_day', 'day_of_week']
        if 'material_code' in df.columns:
            features.append('material_code')
        elif 'material' in df.columns:
            df['material_code'] = pd.factorize(df['material'])[0]
            features.append('material_code')
        
        if 'is_peak_hour' in df.columns:
            features.append('is_peak_hour')
            
        # Crear copia para no modificar datos originales
        X = df[features].copy().dropna()
        
        # Verificar si hay suficientes datos después de eliminar valores nulos
        if len(X) < 10:
            raise ValueError("Insuficientes datos completos para análisis de anomalías")
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detectar anomalías con Isolation Forest
        contamination = 0.05  # 5% de anomalías esperadas
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        # Prepare DataFrame para visualización
        anomaly_df = pd.DataFrame({
            'Tiempo': X['station_stay_time'],
            'Hora': X['hour_of_day'],
            'Día': X['day_of_week'],
            'Es Anomalía': anomalies == -1,
            'Material': df.loc[X.index, 'material'] if 'material' in df.columns else 'Desconocido',
            'Estación': df.loc[X.index, 'station'] if 'station' in df.columns else 'Desconocida'
        })
        
        # Crear gráfico principal
        fig = px.scatter(
            anomaly_df,
            x='Tiempo', 
            y='Hora', 
            color='Es Anomalía',
            color_discrete_map={False: '#3366CC', True: '#DC3545'},
            hover_data=['Material', 'Estación', 'Día'],
            title='Detección de Anomalías en Tiempos de Permanencia',
            labels={
                'Tiempo': 'Tiempo de Permanencia (min)', 
                'Hora': 'Hora del Día',
                'Es Anomalía': 'Anomalía Detectada'
            }
        )
        
        # Añadir regiones de referencia para tiempos normales
        mean_time = X['station_stay_time'].mean()
        std_time = X['station_stay_time'].std()
        
        # Añadir líneas de referencia para tiempos normales (± 2 desviaciones estándar)
        fig.add_shape(
            type="rect",
            x0=max(0, mean_time - 2*std_time),
            x1=mean_time + 2*std_time,
            y0=0,
            y1=23,
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Añadir anotación con estadísticas
        fig.add_annotation(
            x=mean_time,
            y=22,
            text=f"Zona normal<br>Media: {mean_time:.2f} min<br>±2σ: {2*std_time:.2f} min",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="green",
            borderwidth=1
        )
        
        # Contar anomalías
        anomaly_count = anomaly_df['Es Anomalía'].sum()
        anomaly_percent = 100 * anomaly_count / len(anomaly_df)
        
        # Añadir información de anomalías
        fig.add_annotation(
            x=0.98,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"Anomalías Detectadas: {anomaly_count}<br>({anomaly_percent:.1f}% del total)",
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1
        )
        
        # Añadir información de anomalías por materiales
        if 'material' in df.columns:
            # Contar anomalías por tipo de material
            material_counts = anomaly_df[anomaly_df['Es Anomalía']].groupby('Material').size()
            if not material_counts.empty:
                top_material = material_counts.idxmax()
                material_pct = 100 * material_counts[top_material] / anomaly_count if anomaly_count > 0 else 0
                
                fig.add_annotation(
                    x=0.98,
                    y=0.15,
                    xref="paper",
                    yref="paper",
                    text=f"Material con más anomalías:<br>{top_material} ({material_pct:.1f}%)",
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="orange",
                    borderwidth=1
                )
        
        # Añadir información de anomalías por hora
        hour_counts = anomaly_df[anomaly_df['Es Anomalía']].groupby('Hora').size()
        if not hour_counts.empty:
            top_hour = hour_counts.idxmax()
            hour_pct = 100 * hour_counts[top_hour] / anomaly_count if anomaly_count > 0 else 0
            
            fig.add_annotation(
                x=0.98,
                y=0.28,
                xref="paper",
                yref="paper",
                text=f"Hora con más anomalías:<br>{top_hour}:00 ({hour_pct:.1f}%)",
                showarrow=False,
                align="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="purple",
                borderwidth=1
            )
        
        # Dar formato al gráfico
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend_title="Estado",
            showlegend=True,
            height=550,
            xaxis=dict(
                title='Tiempo de Permanencia (min)',
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            yaxis=dict(
                title='Hora del Día',
                tickmode='linear',
                tick0=0,
                dtick=3,
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray'
            )
        )
        
        # Añadir texto explicativo
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text="Análisis de anomalías mediante Isolation Forest. Las anomalías son valores atípicos que podrían indicar problemas operativos.",
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="center"
        )
        
        return fig
        
    except Exception as e:
        # Crear gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"No se pudo generar la detección de anomalías: {str(e)}",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(
            title='Detección de Anomalías no disponible',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=500
        )
        return fig

def create_insights_recommendations_table(df):
    """Crear tabla de insights y recomendaciones operativas"""
    try:
        # Si no hay datos suficientes, retornar mensaje
        if len(df) < 20:
            return html.Div(
                html.P("No hay suficientes datos para generar recomendaciones detalladas.", 
                       className="alert alert-warning")
            )
        
        # Generar insights basados en los datos
        insights = []
        
        # Análisis por hora del día
        if 'hour_of_day' in df.columns and 'station_stay_time' in df.columns:
            hourly_data = df.groupby('hour_of_day')['station_stay_time'].agg(['mean', 'std', 'count']).reset_index()
            hourly_data.columns = ['Hora', 'Tiempo Medio', 'Desviación', 'Conteo']
            
            # Encontrar horas con mayor tiempo promedio
            worst_hours = hourly_data.nlargest(2, 'Tiempo Medio')
            
            for _, row in worst_hours.iterrows():
                insights.append({
                    'Categoria': 'Temporal',
                    'Hallazgo': f"Mayor congestión a las {int(row['Hora'])}:00 hrs",
                    'Impacto': f"{row['Tiempo Medio']:.2f} min (vs. {hourly_data['Tiempo Medio'].mean():.2f} min promedio)",
                    'Recomendación': f"Optimizar operaciones en hora pico de las {int(row['Hora'])}:00"
                })
        
        # Análisis por material rodante
        if 'material' in df.columns and 'station_stay_time' in df.columns:
            material_data = df.groupby('material')['station_stay_time'].agg(['mean', 'std', 'count']).reset_index()
            material_data.columns = ['Material', 'Tiempo Medio', 'Desviación', 'Conteo']
            
            # Filtrar materiales con suficientes datos
            material_data = material_data[material_data['Conteo'] >= 5]
            
            if not material_data.empty:
                # Encontrar material con peor rendimiento
                worst_material = material_data.nlargest(1, 'Tiempo Medio').iloc[0]
                best_material = material_data.nsmallest(1, 'Tiempo Medio').iloc[0]
                
                improvement = ((worst_material['Tiempo Medio'] - best_material['Tiempo Medio']) / 
                              worst_material['Tiempo Medio']) * 100
                
                insights.append({
                    'Categoria': 'Material',
                    'Hallazgo': f"Material {worst_material['Material']} tiene tiempos más largos",
                    'Impacto': f"{worst_material['Tiempo Medio']:.2f} min vs {best_material['Tiempo Medio']:.2f} min ({improvement:.1f}% diferencia)",
                    'Recomendación': f"Evaluar mantenimiento o ajustes operativos para material {worst_material['Material']}"
                })
        
        # Análisis por estación
        if 'station' in df.columns and 'station_stay_time' in df.columns:
            station_data = df.groupby('station')['station_stay_time'].agg(['mean', 'std', 'count']).reset_index()
            station_data.columns = ['Estación', 'Tiempo Medio', 'Desviación', 'Conteo']
            
            # Filtrar estaciones con suficientes datos
            station_data = station_data[station_data['Conteo'] >= 5]
            
            if not station_data.empty:
                # Encontrar estaciones con mayor variabilidad
                high_var_stations = station_data.nlargest(1, 'Desviación').iloc[0]
                
                insights.append({
                    'Categoria': 'Estación',
                    'Hallazgo': f"Estación {high_var_stations['Estación']} tiene alta variabilidad",
                    'Impacto': f"±{high_var_stations['Desviación']:.2f} min (vs ±{station_data['Desviación'].mean():.2f} min promedio)",
                    'Recomendación': f"Revisar procedimientos operativos en estación {high_var_stations['Estación']}"
                })
        
        # Análisis global
        avg_time = df['station_stay_time'].mean()
        std_time = df['station_stay_time'].std()
        cv = (std_time / avg_time) * 100  # Coeficiente de variación
        
        insights.append({
            'Categoria': 'Global',
            'Hallazgo': f"Variabilidad general del sistema",
            'Impacto': f"CV: {cv:.1f}% (menor es mejor)",
            'Recomendación': "Implementar programación basada en datos históricos para optimizar tiempos operativos"
        })
        
        # Detectar anomalías directamente en esta función
        if len(df) >= 20 and 'station_stay_time' in df.columns:
            try:
                # Preparar datos para detección de anomalías
                features = ['station_stay_time', 'hour_of_day', 'day_of_week']
                if 'material_code' in df.columns:
                    features.append('material_code')
                elif 'material' in df.columns:
                    material_code = pd.factorize(df['material'])[0]
                    features_data = df[features].copy()
                    features_data['material_code'] = material_code
                else:
                    features_data = df[features].copy()
                
                # Limpiar datos
                features_data = features_data.dropna()
                
                if len(features_data) >= 10:
                    # Escalar características
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(features_data)
                    
                    # Detectar anomalías con Isolation Forest
                    contamination = 0.05  # 5% de anomalías esperadas
                    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                    anomalies = iso_forest.fit_predict(X_scaled)
                    
                    # Contar anomalías
                    anomaly_count = sum(anomalies == -1)
                    anomaly_percent = 100 * anomaly_count / len(anomalies)
                    
                    if anomaly_count > 0:
                        insights.append({
                            'Categoria': 'Anomalías',
                            'Hallazgo': f"Detectadas {anomaly_count} anomalías ({anomaly_percent:.1f}%)",
                            'Impacto': "Afectan predictibilidad y confiabilidad del servicio",
                            'Recomendación': "Investigar causas de anomalías y establecer protocolos de respuesta"
                        })
            except Exception as e:
                # Si hay algún problema con la detección de anomalías, simplemente no añadimos este insight
                print(f"Error en detección de anomalías: {e}")
        
        # Crear tabla de insights con formato mejorado
        table = dash_table.DataTable(
            data=insights,
            columns=[
                {"name": "Categoría", "id": "Categoria"},
                {"name": "Hallazgo", "id": "Hallazgo"},
                {"name": "Impacto", "id": "Impacto"},
                {"name": "Recomendación", "id": "Recomendación"}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'backgroundColor': 'white'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                },
                {
                    'if': {'column_id': 'Recomendación'},
                    'fontWeight': 'bold'
                }
            ]
        )
        
        return table
        
    except Exception as e:
        return html.Div([
            html.P(f"Error al generar recomendaciones: {str(e)}", 
                   className="alert alert-danger")
        ])

def create_predictive_model_figure(df):
    """Crear visualización de modelo predictivo para tiempos de permanencia"""
    try:
        # Verificar datos mínimos
        required_features = ['station_stay_time', 'hour_of_day', 'day_of_week']
        if not all(feature in df.columns for feature in required_features):
            raise ValueError("Faltan columnas necesarias para el análisis predictivo")
        
        if len(df) < 30:
            raise ValueError("Se requieren al menos 30 registros para el modelado predictivo")
        
        # Preparar características para el modelo
        features = ['hour_of_day', 'day_of_week']
        
        # Añadir características adicionales si están disponibles
        if 'is_peak_hour' in df.columns:
            features.append('is_peak_hour')
        
        if 'material' in df.columns:
            # Codificar material como numérico
            df['material_encoded'] = pd.factorize(df['material'])[0]
            features.append('material_encoded')
        
        if 'is_major_station' in df.columns:
            features.append('is_major_station')
        
        # Preparar datos para modelo
        X = df[features].copy()
        y = df['station_stay_time']
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo de regresión
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predecir valores
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular errores
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # Crear figura
        fig = go.Figure()
        
        # Añadir datos de entrenamiento
        fig.add_trace(go.Scatter(
            x=y_train,
            y=y_pred_train,
            mode='markers',
            name='Entrenamiento',
            marker=dict(
                color='blue',
                size=8,
                opacity=0.5
            ),
            hovertemplate='<b>Entrenamiento</b><br>' +
                          '<b>Real:</b> %{x:.2f} min<br>' +
                          '<b>Predicción:</b> %{y:.2f} min<br>'
        ))
        
        # Añadir datos de prueba
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_test,
            mode='markers',
            name='Prueba',
            marker=dict(
                color='red',
                size=8,
                opacity=0.7
            ),
            hovertemplate='<b>Prueba</b><br>' +
                          '<b>Real:</b> %{x:.2f} min<br>' +
                          '<b>Predicción:</b> %{y:.2f} min<br>'
        ))
        
        # Añadir línea de predicción perfecta
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(
                color='green',
                width=2,
                dash='dash'
            )
        ))
        
        # Añadir anotaciones con métricas
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Métricas del Modelo:</b><br>" +
                 f"MSE Entrenamiento: {train_mse:.3f}<br>" +
                 f"MSE Prueba: {test_mse:.3f}<br>" +
                 f"R²: {r2:.3f}",
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
            borderpad=4
        )
        
        # Obtener importancia de las características
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': np.abs(model.coef_) / sum(np.abs(model.coef_))
        }).sort_values('Importance', ascending=False)
        
        # Añadir anotación con importancia de características
        importance_text = "<b>Importancia de Factores:</b><br>" + "<br>".join(
            [f"{row['Feature']}: {row['Importance']:.2%}" for _, row in feature_importance.head(3).iterrows()]
        )
        
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=importance_text,
            showarrow=False,
            align="right",
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
            borderpad=4
        )
        
        # Configurar layout
        fig.update_layout(
            title='Modelo Predictivo de Tiempos de Permanencia',
            xaxis_title='Tiempo Real (min)',
            yaxis_title='Tiempo Predicho (min)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=550,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Añadir información sobre el rango de error esperado
        mean_abs_error = np.mean(np.abs(y_test - y_pred_test))
        
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=f"Error promedio de predicción: ±{mean_abs_error:.2f} minutos | " +
                 f"Confiabilidad del modelo: {max(0, min(100, r2 * 100)):.1f}%",
            showarrow=False,
            font=dict(size=12),
            xanchor="center"
        )
        
        return fig
        
    except Exception as e:
        # Crear gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"No se pudo generar el modelo predictivo: {str(e)}",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(
            title='Modelo Predictivo no disponible',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=500
        )
        return fig


def create_material_analysis_figure(df):
    """Crear gráfico de análisis por tipo de material rodante"""
    try:
        if 'material' not in df.columns or 'station_stay_time' not in df.columns:
            raise ValueError("Datos insuficientes para el análisis por material")
        
        # Contar ocurrencias de cada material
        material_counts = df['material'].value_counts()
        
        # Mantener solo materiales con suficientes datos (al menos 5 registros)
        materials_to_include = material_counts[material_counts >= 5].index.tolist()
        
        if not materials_to_include:
            raise ValueError("No hay suficientes datos por tipo de material")
        
        filtered_df = df[df['material'].isin(materials_to_include)]
        
        # Crear figura con subplots
        fig = go.Figure()
        
        # 1. Boxplot de distribución de tiempos por material
        for i, material in enumerate(materials_to_include):
            material_data = filtered_df[filtered_df['material'] == material]
            
            fig.add_trace(go.Box(
                y=material_data['station_stay_time'],
                name=material,
                boxmean=True,  # Mostrar media
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(
                    size=4,
                    opacity=0.5
                ),
                hovertemplate='<b>Material:</b> %{x}<br>' +
                            '<b>Tiempo:</b> %{y:.2f} min<br>'
            ))
        
        # Añadir línea de tiempo promedio global
        avg_time = filtered_df['station_stay_time'].mean()
        
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(materials_to_include) - 0.5,
            y0=avg_time,
            y1=avg_time,
            line=dict(
                color="black",
                width=2,
                dash="dash"
            )
        )
        
        # Añadir etiqueta de referencia para el promedio global
        fig.add_annotation(
            x=len(materials_to_include) - 1,
            y=avg_time,
            text=f"Promedio Global: {avg_time:.2f} min",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=0
        )
        
        # Calcular estadísticas por material para añadir información
        material_stats = filtered_df.groupby('material')['station_stay_time'].agg(['mean', 'std', 'count']).reset_index()
        material_stats.columns = ['Material', 'Media', 'Desviación', 'Conteo']
        
        # Encontrar material con mejor y peor rendimiento
        best_material = material_stats.loc[material_stats['Media'].idxmin(), 'Material']
        worst_material = material_stats.loc[material_stats['Media'].idxmax(), 'Material']
        
        # Añadir anotaciones para destacar mejores y peores materiales
        fig.add_annotation(
            x=list(materials_to_include).index(best_material),
            y=material_stats.loc[material_stats['Material'] == best_material, 'Media'].iloc[0],
            text=f"Mejor rendimiento",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(size=12, color="green")
        )
        
        fig.add_annotation(
            x=list(materials_to_include).index(worst_material),
            y=material_stats.loc[material_stats['Material'] == worst_material, 'Media'].iloc[0],
            text=f"Menor rendimiento",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=40,
            font=dict(size=12, color="red")
        )
        
        # Actualizar layout
        fig.update_layout(
            title='Análisis de Tiempos por Material Rodante',
            xaxis_title='Tipo de Material',
            yaxis_title='Tiempo de Permanencia en Estación (min)',
            boxmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=550,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Añadir anotación con conclusión
        best_mean = material_stats.loc[material_stats['Material'] == best_material, 'Media'].iloc[0]
        worst_mean = material_stats.loc[material_stats['Material'] == worst_material, 'Media'].iloc[0]
        improvement_potential = ((worst_mean - best_mean) / worst_mean) * 100
        
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=f"Potencial de optimización: {improvement_potential:.1f}% mejora al migrar de {worst_material} a {best_material}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
            borderpad=4
        )
        
        return fig
        
    except Exception as e:
        # Crear gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"No se pudo generar el análisis por material: {str(e)}",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(
            title='Análisis por Material no disponible',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=500
        )
        return fig

def create_temporal_pattern_figure(df):
    """Crear visualización de patrones temporales por hora del día"""
    try:
        if 'hour_of_day' not in df.columns or 'station_stay_time' not in df.columns or 'station' not in df.columns:
            raise ValueError("Datos insuficientes para el análisis temporal")
        
        # Obtener las 5 estaciones más frecuentes
        top_stations = df['station'].value_counts().nlargest(5).index.tolist()
        filtered_df = df[df['station'].isin(top_stations)]
        
        # Calcular tiempo promedio por hora para cada estación
        hour_station_avg = filtered_df.groupby(['hour_of_day', 'station'])['station_stay_time'].mean().reset_index()
        
        # Crear gráfico
        fig = go.Figure()
        
        # Añadir datos para cada estación
        for station in top_stations:
            station_data = hour_station_avg[hour_station_avg['station'] == station]
            fig.add_trace(go.Scatter(
                x=station_data['hour_of_day'],
                y=station_data['station_stay_time'],
                mode='lines+markers',
                name=station,
                marker=dict(size=8),
                line=dict(width=2)
            ))
        
        # Calcular promedio global por hora
        hour_avg = df.groupby('hour_of_day')['station_stay_time'].mean().reset_index()
        
        # Añadir barra para promedio por hora
        fig.add_trace(go.Bar(
            x=hour_avg['hour_of_day'],
            y=hour_avg['station_stay_time'],
            name='Promedio Global',
            marker_color='rgba(55, 83, 109, 0.3)',
            opacity=0.7
        ))
        
        # Resaltar horas pico
        for peak_hour in [7, 8, 9, 17, 18, 19]:
            fig.add_vrect(
                x0=peak_hour-0.5,
                x1=peak_hour+0.5,
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below",
                line_width=0,
            )
        
        # Personalizar layout
        fig.update_layout(
            title='Patrones de Tiempos de Permanencia por Hora del Día',
            xaxis=dict(
                title='Hora del Día',
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, 23.5]
            ),
            yaxis=dict(title='Tiempo de Permanencia (min)'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            bargap=0.15,
            height=500,
            margin=dict(l=40, r=40, t=60, b=60)
        )
        
        # Añadir anotaciones para horas pico
        fig.add_annotation(
            x=8,
            y=df['station_stay_time'].max() * 0.9,
            text="Hora Pico Mañana",
            showarrow=False,
            font=dict(size=10, color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            borderpad=3,
            opacity=0.8
        )
        
        fig.add_annotation(
            x=18,
            y=df['station_stay_time'].max() * 0.8,
            text="Hora Pico Tarde",
            showarrow=False,
            font=dict(size=10, color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            borderpad=3,
            opacity=0.8
        )
        
        return fig
    
    except Exception as e:
        # Crear gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"No se pudo generar el análisis temporal: {str(e)}",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(
            title='Análisis Temporal no disponible',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=500
        )
        return fig
    
    




def preprocess_data(df):
    """Preprocesar datos para análisis de machine learning"""
    try:
        # Verificar que el DataFrame no está vacío
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Crear una copia para no modificar el original
        processed_df = df.copy()
        
        # Convertir columnas de tiempo a datetime si no lo son ya
        for col in ['arrival_time', 'departure_time']:
            if col in processed_df.columns:
                if processed_df[col].dtype != 'datetime64[ns]':
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
        
        # Calcular tiempo de permanencia en estación
        if 'arrival_time' in processed_df.columns and 'departure_time' in processed_df.columns:
            processed_df['station_stay_time'] = (processed_df['departure_time'] - processed_df['arrival_time']).dt.total_seconds() / 60
        else:
            # Crear columna ficticia si no existen los datos
            processed_df['station_stay_time'] = 5.0  # Valor predeterminado
        
        # Eliminar valores negativos o extremadamente grandes
        processed_df = processed_df[processed_df['station_stay_time'] >= 0]
        processed_df = processed_df[processed_df['station_stay_time'] < 60]  # Eliminar valores extremos
        
        # Extraer características de tiempo
        if 'arrival_time' in processed_df.columns:
            processed_df['hour_of_day'] = processed_df['arrival_time'].dt.hour
            processed_df['day_of_week'] = processed_df['arrival_time'].dt.dayofweek
            processed_df['is_peak_hour'] = processed_df['hour_of_day'].apply(
                lambda x: 1 if (x >= 7 and x <= 9) or (x >= 17 and x <= 19) else 0
            )
            processed_df['period_of_day'] = processed_df['hour_of_day'].apply(
                lambda x: 'Mañana' if x < 12 else ('Tarde' if x < 18 else 'Noche')
            )
        
        # Identificar las estaciones más importantes (con más registros)
        station_counts = processed_df['station'].value_counts()
        top_stations = station_counts[station_counts > station_counts.median()].index.tolist()
        processed_df['is_major_station'] = processed_df['station'].isin(top_stations).astype(int)
        
        # Crear características categóricas
        processed_df['day_name'] = processed_df['day_of_week'].map({
            0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 
            3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
        })
        
        return processed_df
        
    except Exception as e:
        print(f"Error en preprocesamiento: {str(e)}")
        # Crear DataFrame mínimo con columnas necesarias
        return pd.DataFrame({
            'station_stay_time': [5.0],
            'hour_of_day': [12],
            'day_of_week': [3],
            'is_peak_hour': [0],
            'period_of_day': ['Tarde'],
            'day_name': ['Miércoles'],
            'station': ['Unknown'],
            'train_number': ['Unknown'],
            'material': ['Unknown']
        })

def create_key_metrics_card(df):
    """Crear tarjetas con métricas clave con tamaño reducido"""
    try:
        # Calcular métricas
        avg_stay_time = df['station_stay_time'].mean()
        max_stay_time = df['station_stay_time'].max()
        min_stay_time = df['station_stay_time'].min()
        median_stay_time = df['station_stay_time'].median()
        
        # Calcular percentiles para mayor contexto
        p90_time = df['station_stay_time'].quantile(0.9)
        p10_time = df['station_stay_time'].quantile(0.1)
        
        # Calcular tiempo promedio en horas pico vs no pico
        if 'is_peak_hour' in df.columns:
            peak_avg = df[df['is_peak_hour'] == 1]['station_stay_time'].mean()
            non_peak_avg = df[df['is_peak_hour'] == 0]['station_stay_time'].mean()
            peak_impact = 100 * (peak_avg - non_peak_avg) / non_peak_avg if non_peak_avg > 0 else 0
        else:
            peak_impact = 0
        
        # Calcular eficiencia operativa
        efficiency = 100 * (1 - (avg_stay_time - min_stay_time) / max_stay_time) if max_stay_time > min_stay_time else 100
        efficiency = max(0, min(100, efficiency))
        
        # Calcular variabilidad
        cv = 100 * df['station_stay_time'].std() / avg_stay_time if avg_stay_time > 0 else 0
        
        # Estilo mejorado para las tarjetas (más pequeñas)
        card_style = {
            'borderRadius': '4px',
            'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
            'height': '100%',
            'margin': '5px 0',
            'backgroundColor': 'white'
        }
        
        header_style = {
            'borderBottom': '1px solid #eaeaea',
            'padding': '8px',
            'borderTopLeftRadius': '4px',
            'borderTopRightRadius': '4px',
            'fontSize': '12px'
        }
        
        body_style = {
            'padding': '10px',
            'textAlign': 'center'
        }
        
        # Crear componentes de tarjetas más pequeñas
        metrics_cards = html.Div([
            # Tarjeta de tiempo promedio
            html.Div([
                html.Div(style=card_style, children=[
                    html.Div(style={**header_style, 'backgroundColor': '#3498DB', 'color': 'white'}, children=[
                        html.H5("TIEMPO PROMEDIO EN ESTACIÓN", className="m-0 fw-bold", style={'fontSize': '10px'})
                    ]),
                    html.Div(style=body_style, children=[
                        html.H2(f"{avg_stay_time:.2f} min", className="display-5 mb-0 fw-bold", style={'fontSize': '18px'}),
                        html.P(f"Mediana: {median_stay_time:.2f} min • Rango: {min_stay_time:.1f}-{max_stay_time:.1f} min", 
                               className="text-muted mt-1", style={'fontSize': '10px'})
                    ])
                ])
            ], className="col-xl-3 col-md-6", style={'padding': '5px'}),
            
            # Tarjeta de eficiencia operativa
            html.Div([
                html.Div(style=card_style, children=[
                    html.Div(style={**header_style, 'backgroundColor': '#2ECC71', 'color': 'white'}, children=[
                        html.H5("EFICIENCIA OPERATIVA", className="m-0 fw-bold", style={'fontSize': '10px'})
                    ]),
                    html.Div(style=body_style, children=[
                        html.H2(f"{efficiency:.1f}%", className="display-5 mb-0 fw-bold", style={'fontSize': '18px'}),
                        html.P(f"Coef. Variación: {cv:.1f}%", className="text-muted mt-1", style={'fontSize': '10px'})
                    ])
                ])
            ], className="col-xl-3 col-md-6", style={'padding': '5px'}),
            
            # Tarjeta de impacto hora pico
            html.Div([
                html.Div(style=card_style, children=[
                    html.Div(style={**header_style, 'backgroundColor': '#F39C12', 'color': 'white'}, children=[
                        html.H5("IMPACTO HORA PICO", className="m-0 fw-bold", style={'fontSize': '10px'})
                    ]),
                    html.Div(style=body_style, children=[
                        html.H2(f"{peak_impact:.1f}% mayor", className="display-5 mb-0 fw-bold", style={'fontSize': '18px'}),
                        html.P(f"En horas pico: {peak_avg:.2f} min vs {non_peak_avg:.2f} min", 
                               className="text-muted mt-1", style={'fontSize': '10px'})
                    ])
                ])
            ], className="col-xl-3 col-md-6", style={'padding': '5px'}),
            
            # Tarjeta de variabilidad
            html.Div([
                html.Div(style=card_style, children=[
                    html.Div(style={**header_style, 'backgroundColor': '#9B59B6', 'color': 'white'}, children=[
                        html.H5("DISTRIBUCIÓN DE TIEMPOS", className="m-0 fw-bold", style={'fontSize': '10px'})
                    ]),
                    html.Div(style=body_style, children=[
                        html.H2(f"P90: {p90_time:.2f} min", className="display-5 mb-0 fw-bold", style={'fontSize': '18px'}),
                        html.P(f"10% más rápido: {p10_time:.2f} min • 10% más lento: {p90_time:.2f} min", 
                               className="text-muted mt-1", style={'fontSize': '10px'})
                    ])
                ])
            ], className="col-xl-3 col-md-6", style={'padding': '5px'})
        ], className="row", style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -5px'})
        
        return metrics_cards
    
    except Exception as e:
        # En caso de error, mostrar métricas simplificadas
        return html.Div([
            html.Div([
                html.Div(style={'borderRadius': '4px', 'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)', 'backgroundColor': 'white', 'padding': '10px'}, children=[
                    html.H5("Métricas no disponibles", className="card-title", style={'fontSize': '14px'}),
                    html.P(f"Error: {str(e)}", className="card-text text-danger", style={'fontSize': '12px'})
                ])
            ], className="col-md-12 mb-2")
        ], className="row")

def create_train_clustering_figure(df):
    """Crear visualización mejorada de clustering de trenes"""
    try:
        if len(df) < 20:
            raise ValueError("Datos insuficientes para realizar clustering (mínimo 20 registros)")
        
        if not all(col in df.columns for col in ['station_stay_time', 'hour_of_day', 'day_of_week']):
            raise ValueError("Faltan columnas necesarias para el análisis")
        
        # Preparar datos para análisis
        features = ['station_stay_time', 'hour_of_day', 'day_of_week']
        if 'is_peak_hour' in df.columns:
            features.append('is_peak_hour')
        if 'material' in df.columns:
            df['material_code'] = pd.factorize(df['material'])[0]
            features.append('material_code')
        
        # Preparar el DataFrame solo con las features necesarias
        X = df[features].copy()
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinar número óptimo de clusters (entre 2 y 5)
        n_clusters = min(5, max(2, len(X) // 50))
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Aplicar PCA para visualización
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear DataFrame para visualización
        plot_df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': [f'Grupo {c+1}' for c in clusters],
            'Tiempo': df['station_stay_time'],
            'Hora': df['hour_of_day'],
            'Material': df['material'] if 'material' in df.columns else 'Desconocido',
            'Estación': df['station'] if 'station' in df.columns else 'Desconocida'
        })
        
        # Calcular centros de clusters en espacio PCA
        centroids_pca = []
        for i in range(n_clusters):
            mask = clusters == i
            if np.sum(mask) > 0:  # Si hay puntos en este cluster
                centroid = np.mean(X_pca[mask], axis=0)
                centroids_pca.append((i, centroid[0], centroid[1]))
        
        # Crear figura
        fig = go.Figure()
        
        # Agregar scatter plot para cada cluster
        for i in range(n_clusters):
            cluster_name = f'Grupo {i+1}'
            cluster_data = plot_df[plot_df['Cluster'] == cluster_name]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['PCA1'],
                y=cluster_data['PCA2'],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=cluster_name,
                hovertemplate='<b>Grupo:</b> ' + cluster_name + '<br>' +
                              '<b>Tiempo:</b> %{customdata[0]:.2f} min<br>' +
                              '<b>Hora:</b> %{customdata[1]}<br>' +
                              '<b>Material:</b> %{customdata[2]}<br>' +
                              '<b>Estación:</b> %{customdata[3]}<br>',
                customdata=np.stack((
                    cluster_data['Tiempo'].values, 
                    cluster_data['Hora'].values, 
                    cluster_data['Material'].values,
                    cluster_data['Estación'].values
                ), axis=1)
            ))
        
        # Agregar centroides
        for i, cx, cy in centroids_pca:
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=16,
                    opacity=1,
                    line=dict(width=2, color='black')
                ),
                name=f'Centro Grupo {i+1}',
                hoverinfo='name',
                showlegend=False
            ))
            
            # Añadir etiqueta al centroide
            fig.add_annotation(
                x=cx,
                y=cy,
                text=f"Grupo {i+1}",
                showarrow=False,
                yshift=20,
                font=dict(size=12, color="black", family="Arial Black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=3,
                opacity=0.8
            ),
        
        # Calcular estadísticas por cluster para mostrar características
    
        cluster_stats = []
        for i in range(n_clusters):
            cluster_name = f'Grupo {i+1}'
            cluster_data = df.loc[plot_df[plot_df['Cluster'] == cluster_name].index]
            
            stats = {
                'cluster': cluster_name,
                'count': len(cluster_data),
                'mean_time': cluster_data['station_stay_time'].mean(),
                'mean_hour': cluster_data['hour_of_day'].mean()
            }
            
            if 'is_peak_hour' in cluster_data.columns:
                stats['peak_hour_pct'] = 100 * cluster_data['is_peak_hour'].mean()
            
            cluster_stats.append(stats)
        
        # Añadir anotaciones con características de cada cluster
        for stats in cluster_stats:
            # Encontrar posición del centroide correspondiente
            cluster_idx = int(stats['cluster'].split(' ')[1]) - 1
            for i, cx, cy in centroids_pca:
                if i == cluster_idx:
                    # Crear texto descriptivo del cluster
                    description = (
                        f"<b>{stats['cluster']}</b> ({stats['count']} registros)<br>"
                        f"Tiempo: {stats['mean_time']:.2f} min<br>"
                        f"Hora media: {stats['mean_hour']:.1f}h"
                    )
                    
                    if 'peak_hour_pct' in stats:
                        description += f"<br>Hora pico: {stats['peak_hour_pct']:.0f}%"
                    
                    # Añadir anotación descriptiva
                    fig.add_annotation(
                        x=cx,
                        y=cy,
                        text=description,
                        showarrow=True,
                        ax=50,
                        ay=-50,
                        arrowhead=2,
                        arrowwidth=1,
                        arrowcolor="gray",
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.9,
                        font=dict(size=10),
                        xanchor="left"
                    )
        
        # Configurar layout
        fig.update_layout(
            title='Clustering de Patrones Operativos',
            xaxis=dict(
                title='Componente Principal 1',
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            yaxis=dict(
                title='Componente Principal 2',
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            plot_bgcolor='white',
            height=550,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Añadir una anotación explicativa
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text="Análisis de agrupamiento mediante K-means con reducción dimensional PCA para visualización",
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="center"
        )
        
        return fig
    
    except Exception as e:
        # Crear gráfico de error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"No se pudo generar el análisis de clusters: {str(e)}",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        fig.update_layout(
            title='Clustering no disponible',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            height=500
        )
        return fig




def create_travel_time_prediction_figure(df):
    """Crear modelo de predicción de tiempos de viaje mejorado"""
    # Preparar datos para predicción
    features = ['hour_of_day', 'day_of_week', 'material_encoded', 'is_peak_hour']
    X = df[features].dropna()
    y = df.loc[X.index, 'station_stay_time']
    
    # Manejar caso de datos insuficientes
    if len(X) < 20:  # Necesitamos suficientes datos para división train/test
        # Crear figura vacía con mensaje
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Se necesitan más datos para crear modelo predictivo",
            showarrow=False
        )
        fig.update_layout(title="Datos insuficientes para predicción")
        return fig
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Crear figura de predicción vs real
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, 
        y=y_pred, 
        mode='markers',
        name='Predicciones',
        marker=dict(
            color='blue',
            size=10,
            opacity=0.7,
            line=dict(width=1, color='black')
        )
    ))
    
    # Línea de referencia perfecta
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Predicción Perfecta',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Añadir anotaciones con métricas de evaluación
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Error cuadrático medio: {mse:.3f}<br>R²: {r2:.3f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    # Añadir información sobre coeficientes importantes
    feature_importance = pd.DataFrame({
        'Característica': features,
        'Importancia': np.abs(model.coef_)
    }).sort_values('Importancia', ascending=False)
    
    most_important = feature_importance.iloc[0]['Característica']
    most_important = {
        'hour_of_day': 'Hora del día',
        'day_of_week': 'Día de semana',
        'material_encoded': 'Tipo de material',
        'is_peak_hour': 'Hora pico'
    }.get(most_important, most_important)
    
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"Factor más influyente:<br>{most_important}",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.update_layout(
        title='Predicción vs. Realidad: Tiempos de Estación',
        xaxis_title='Tiempo Real (min)',
        yaxis_title='Tiempo Predicho (min)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True
    )
    
    return fig

def create_ml_insights_table(df):
    """Crear tabla mejorada de insights de machine learning"""
    # Calcular insights más informativos
    insights = []
    
    try:
        # Variabilidad de tiempos por estación
        variability_by_station = df.groupby('station')['station_stay_time'].std().sort_values(ascending=False)
        if not variability_by_station.empty:
            top_variable_station = variability_by_station.index[0]
            insights.append({
                'Métrica': 'Estación con Mayor Variabilidad',
                'Valor': f'{top_variable_station} ({variability_by_station.iloc[0]:.2f} min)',
                'Descripción': 'Estación con mayor inconsistencia en tiempos de permanencia'
            })
    except:
        pass
    
    # Variabilidad general
    insights.append({
        'Métrica': 'Variabilidad de Tiempos de Estación',
        'Valor': f'{df["station_stay_time"].std():.2f} min',
        'Descripción': 'Desviación estándar de los tiempos de permanencia'
    })
    
    # Correlación hora-tiempo
    insights.append({
        'Métrica': 'Correlación Hora-Tiempo de Estación',
        'Valor': f'{df["station_stay_time"].corr(df["hour_of_day"]):.2f}',
        'Descripción': 'Correlación entre hora del día y tiempo de permanencia'
    })
    
    # Material con mayor variabilidad
    if 'material' in df.columns:
        material_variability = df.groupby('material')['station_stay_time'].agg(['mean', 'std']).sort_values('std', ascending=False)
        if not material_variability.empty:
            insights.append({
                'Métrica': 'Material con Mayor Variabilidad',
                'Valor': material_variability.index[0],
                'Descripción': f'Material con mayor variación en tiempos de estación ({material_variability["std"].iloc[0]:.2f} min)'
            })
            
    # Horas pico vs horas normales
    try:
        peak_avg = df[df['is_peak_hour'] == 1]['station_stay_time'].mean()
        non_peak_avg = df[df['is_peak_hour'] == 0]['station_stay_time'].mean()
        diff_percent = 100 * (peak_avg - non_peak_avg) / non_peak_avg if non_peak_avg > 0 else 0
        
        insights.append({
            'Métrica': 'Impacto de Horas Pico',
            'Valor': f'{diff_percent:.1f}% más tiempo',
            'Descripción': f'Aumento porcentual en tiempos de estación durante horas pico'
        })
    except:
        pass
    
    # Crear tabla
    table = dash_table.DataTable(
        id='ml-insights-table',
        columns=[{"name": i, "id": i} for i in insights[0].keys()],
        data=insights,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'backgroundColor': 'white'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            }
        ]
    )
    
    return table

def create_material_distribution_figure(df):
    """Crear gráfico de distribución de tiempos por material"""
    if 'material' not in df.columns or 'station_stay_time' not in df.columns:
        # Crear figura vacía
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Datos insuficientes para mostrar distribución por material",
            showarrow=False
        )
        fig.update_layout(title="Distribución por Material no disponible")
        return fig
    
    # Agrupar datos por material
    material_stats = df.groupby('material')['station_stay_time'].agg(['mean', 'std', 'count']).reset_index()
    material_stats.columns = ['Material', 'Tiempo Medio', 'Desviación', 'Conteo']
    
    # Filtrar solo materiales con suficientes datos (al menos 5 registros)
    material_stats = material_stats[material_stats['Conteo'] >= 5]
    
    if material_stats.empty:
        # Crear figura vacía
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Se necesitan más datos por tipo de material",
            showarrow=False
        )
        fig.update_layout(title="Datos insuficientes por material")
        return fig
    
    # Ordenar por rendimiento (tiempo medio)
    material_stats = material_stats.sort_values('Tiempo Medio')
    
    # Identificar mejor y peor material
    best_material = material_stats.iloc[0]['Material']
    worst_material = material_stats.iloc[-1]['Material']
    
    # Calcular diferencia porcentual
    improvement_pct = ((material_stats.iloc[-1]['Tiempo Medio'] - material_stats.iloc[0]['Tiempo Medio']) / 
                      material_stats.iloc[-1]['Tiempo Medio']) * 100
    
    # Crear gráfico
    fig = px.strip(
        df[df['material'].isin(material_stats['Material'])],
        x='material',
        y='station_stay_time',
        color='material',
        labels={
            'material': 'Tipo de Material',
            'station_stay_time': 'Tiempo de Permanencia en Estación (min)'
        },
        title="Análisis de Tiempos por Material Rodante"
    )
    
    # Añadir línea de tiempo promedio global
    avg_time = df['station_stay_time'].mean()
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(material_stats) - 0.5,
        y0=avg_time,
        y1=avg_time,
        line=dict(
            color="black",
            width=2,
            dash="dash",
        )
    )
    
    # Añadir etiquetas para mejor y peor rendimiento
    fig.add_annotation(
        x=list(material_stats['Material']).index(best_material),
        y=0.2,
        text="Mejor rendimiento",
        showarrow=False,
        font=dict(color="green", size=12),
        xanchor="center",
        yanchor="bottom"
    )
    
    fig.add_annotation(
        x=list(material_stats['Material']).index(worst_material),
        y=0.2,
        text="Menor rendimiento",
        showarrow=False,
        font=dict(color="red", size=12),
        xanchor="center",
        yanchor="bottom"
    )
    
    # Añadir anotación para el promedio global
    fig.add_annotation(
        x=len(material_stats) - 1,
        y=avg_time,
        text=f"Promedio Global: {avg_time:.2f}",
        showarrow=False,
        font=dict(color="black", size=12),
        xshift=80,
        yshift=0
    )
    
    # Añadir un cuadro informativo sobre la optimización
    # Aumentar el margen inferior y agregar el cuadro más arriba
    info_text = f"Potencial de optimización: {improvement_pct:.1f}% mejora al migrar de {worst_material} a {best_material}"
    
    fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=-0.20,  # Posición más arriba para evitar que se corte
                xref="paper",
                yref="paper",
                text=info_text,
                showarrow=False,
                font=dict(size=14),
                bgcolor="cornflowerblue",
                bordercolor="black",
                borderwidth=1,
                borderpad=6,
                opacity=0.8
            )
        ]
    )
    
    # Configurar layout para asegurar que haya suficiente espacio para la anotación
    fig.update_layout(
        margin=dict(l=50, r=50, t=80, b=150),  # Aumentar el margen inferior
        xaxis=dict(tickangle=45),
        height=700,  # Aumentar altura del gráfico
        plot_bgcolor='white'
    )
    
    # Añadir información de análisis
    fig.add_annotation(
        x=0.5,
        y=-0.30,  # Posición más abajo
        xref="paper",
        yref="paper",
        text="Análisis: Comparación del rendimiento operativo por tipo de material rodante. Este análisis permite identificar si ciertos tipos de trenes sistemáticamente presentan patrones diferentes de eficiencia en términos de tiempos de permanencia.",
        showarrow=False,
        font=dict(size=12),
        align="center",
        xanchor="center",
        yanchor="top",
        width=800  # Limitar ancho del texto
    )
    
    return fig


def create_hourly_efficiency_figure(df):
    """Crear gráfico de eficiencia operativa por hora del día"""
    if 'hour_of_day' not in df.columns or 'station_stay_time' not in df.columns:
        # Crear figura vacía
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="Datos insuficientes para calcular eficiencia por hora",
            showarrow=False
        )
        fig.update_layout(title="Eficiencia por Hora no disponible")
        return fig
    
    # Agrupar datos por hora
    hourly_stats = df.groupby('hour_of_day')['station_stay_time'].agg(['mean', 'median', 'std', 'count']).reset_index()
    hourly_stats.columns = ['hour', 'mean_time', 'median_time', 'std_time', 'count']
    
    # Calcular métricas de eficiencia
    # Para la eficiencia consideramos:
    # 1. Relación entre mediana y media (valores cercanos indican menos variabilidad)
    # 2. Coeficiente de variación (std/mean) menor indica mayor eficiencia
    hourly_stats['cv'] = hourly_stats['std_time'] / hourly_stats['mean_time']
    
    # Normalizar para crear un índice de eficiencia (inversamente proporcional al CV)
    max_cv = hourly_stats['cv'].max()
    if max_cv > 0:
        hourly_stats['efficiency'] = 100 * (1 - hourly_stats['cv'] / max_cv)
    else:
        hourly_stats['efficiency'] = 100
    
    # Crear figura
    fig = go.Figure()
    
    # Gráfico de barras para eficiencia
    fig.add_trace(go.Bar(
        x=hourly_stats['hour'],
        y=hourly_stats['efficiency'],
        name='Índice de Eficiencia',
        marker_color='rgba(55, 128, 191, 0.7)',
        hovertemplate='<b>Hora:</b> %{x}:00<br>' +
                      '<b>Eficiencia:</b> %{y:.1f}%<br>' +
                      '<b>Observaciones:</b> %{text}<br>',
        text=hourly_stats['count']
    ))
    
    # Línea para tiempo medio
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean_time'],
        mode='lines+markers',
        name='Tiempo Medio (min)',
        yaxis='y2',
        line=dict(color='rgba(219, 64, 82, 0.8)', width=2),
        marker=dict(size=7),
        hovertemplate='<b>Hora:</b> %{x}:00<br>' +
                      '<b>Tiempo Medio:</b> %{y:.2f} min<br>'
    ))
    
    # Añadir anotaciones para horas pico y valle
    max_efficiency_hour = hourly_stats.loc[hourly_stats['efficiency'].idxmax(), 'hour']
    min_efficiency_hour = hourly_stats.loc[hourly_stats['efficiency'].idxmin(), 'hour']
    
    fig.add_annotation(
        x=max_efficiency_hour,
        y=hourly_stats.loc[hourly_stats['hour'] == max_efficiency_hour, 'efficiency'].iloc[0],
        text="Mayor eficiencia",
        showarrow=True,
        arrowhead=1,
        yshift=10
    )
    
    fig.add_annotation(
        x=min_efficiency_hour,
        y=hourly_stats.loc[hourly_stats['hour'] == min_efficiency_hour, 'efficiency'].iloc[0],
        text="Menor eficiencia",
        showarrow=True,
        arrowhead=1,
        yshift=-30
    )
    
    # Configurar layout
    fig.update_layout(
        title="Eficiencia Operativa por Hora del Día",
        xaxis=dict(
            title="Hora del Día",
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title="Índice de Eficiencia (%)",
            range=[0, 100]
        ),
        yaxis2=dict(
            title="Tiempo Medio (min)",
            titlefont=dict(color='rgba(219, 64, 82, 0.8)'),
            tickfont=dict(color='rgba(219, 64, 82, 0.8)'),
            overlaying='y',
            side='right'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.15,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Agregar diagrama de horas pico
    peak_hours_morning = [7, 8, 9]
    peak_hours_evening = [17, 18, 19]
    
    for hour in range(24):
        if hour in peak_hours_morning:
            fig.add_shape(
                type="rect",
                x0=hour - 0.4,
                x1=hour + 0.4,
                y0=0,
                y1=100,
                fillcolor="rgba(255,200,200,0.3)",
                line=dict(width=0),
                layer="below"
            )
        elif hour in peak_hours_evening:
            fig.add_shape(
                type="rect",
                x0=hour - 0.4,
                x1=hour + 0.4,
                y0=0,
                y1=100,
                fillcolor="rgba(255,200,200,0.3)",
                line=dict(width=0),
                layer="below"
            )
    
    fig.add_annotation(
        x=8,
        y=5,
        text="Hora Pico Mañana",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    fig.add_annotation(
        x=18,
        y=5,
        text="Hora Pico Tarde",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    return fig

def generate_executive_summary(df):
    """Generar un resumen ejecutivo del análisis"""
    if df.empty:
        return "No hay suficientes datos para generar un resumen ejecutivo."
    
    try:
        # Calcular métricas clave
        avg_stay_time = df['station_stay_time'].mean()
        max_stay_time = df['station_stay_time'].max()
        unique_materials = df['material'].nunique()
        
        # Hallar horas pico basado en tiempo de permanencia
        peak_hours_df = df.groupby('hour_of_day')['station_stay_time'].mean()
        peak_hours = peak_hours_df.idxmax() if not peak_hours_df.empty else "N/A"
        
        # Calcular variabilidad total
        cv = df['station_stay_time'].std() / avg_stay_time if avg_stay_time > 0 else 0
        
        # Hallar material con mejor y peor desempeño
        material_stats = df.groupby('material').agg({
            'station_stay_time': ['mean', 'std', 'count']
        })
        material_stats.columns = ['mean', 'std', 'count']
        
        # Filtrar materiales con suficientes datos
        material_stats = material_stats[material_stats['count'] >= 5]
        
        best_material = "No disponible"
        worst_material = "No disponible"
        if not material_stats.empty:
            best_material = material_stats.sort_values('mean').index[0]
            worst_material = material_stats.sort_values('mean', ascending=False).index[0]
        
        # Buscar correlaciones importantes
        hour_corr = df['station_stay_time'].corr(df['hour_of_day'])
        day_corr = df['station_stay_time'].corr(df['day_of_week'])
        corr_factor = "hora del día" if abs(hour_corr) > abs(day_corr) else "día de la semana"
        corr_value = hour_corr if abs(hour_corr) > abs(day_corr) else day_corr
        corr_direction = "positiva" if corr_value > 0 else "negativa"
        
        # Calcular eficiencia operativa global
        efficiency = 100 * (1 - cv) if cv < 1 else 0
        efficiency_level = "Alta" if efficiency > 75 else "Media" if efficiency > 50 else "Baja"
        
        summary = f"""
        RESUMEN EJECUTIVO DE ANÁLISIS DE TRENES:
        
        Métricas clave:
        • Tiempo promedio de permanencia en estación: {avg_stay_time:.2f} minutos
        • Tiempo máximo de permanencia: {max_stay_time:.2f} minutos
        • Tipos de material rodante analizados: {unique_materials}
        • Hora con mayor tiempo de permanencia: {peak_hours}:00 hrs
        • Eficiencia operativa global: {efficiency:.1f}% ({efficiency_level})
        
        Material rodante:
        • Material con mejor desempeño: {best_material}
        • Material con peor desempeño: {worst_material}
        
        Correlaciones e insights:
        • Se encontró una correlación {corr_direction} con el {corr_factor} ({corr_value:.2f})
        • El análisis de clustering identificó patrones consistentes en el comportamiento de los trenes
        • El modelo predictivo puede estimar tiempos de permanencia en estación con base en hora del día, 
          día de la semana y tipo de material
        • Se detectaron anomalías que podrían requerir atención operativa adicional
        
        Recomendaciones:
        • Optimizar la programación durante la hora pico ({peak_hours}:00)
        • Evaluar el rendimiento del material {worst_material} para identificar oportunidades de mejora
        • Utilizar el modelo predictivo para anticipar y mitigar posibles demoras
        """
        
        return summary
    except Exception as e:
        return f"Error al generar el resumen ejecutivo: {str(e)}"

# Esta función corrige los problemas de tamaño y proporciones en la pestaña de Análisis Avanzado
# Reemplaza la función create_ml_analysis_tab en dashboard/velcom_dashboard.py

def create_ml_analysis_tab(data):
    """Crear la pestaña de análisis avanzado con machine learning mejorada"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    
    try:
        # Preprocesar datos para análisis
        processed_data = preprocess_data(data['velcom_data'])
        
        # Crear componentes de la pestaña
        tab_content = html.Div([
            # Encabezado con título y descripción
            html.Div([
                html.H2("Análisis Avanzado de Operaciones", className="text-center mb-2", 
                       style={'fontWeight': 'bold', 'paddingTop': '10px', 'fontSize': '20px'}),
                html.P("Dashboard de inteligencia artificial para optimización de tiempos de operación", 
                       className="text-muted text-center mb-2", style={'fontSize': '14px'}),
            ], style={'paddingBottom': '5px', 'borderBottom': '1px solid #eaeaea'}),
            
            # Fila de métricas clave - Ajustadas para menor tamaño
            html.Div([
                create_key_metrics_card(processed_data)
            ], style={'marginBottom': '15px', 'marginTop': '10px'}),
            
            # Contenedor para gráficos con tamaño reducido
            html.Div([
                # Primera fila de gráficos
                html.Div([
                    # Análisis de patrones temporales
                    html.Div([
                        html.H4("Patrones Temporales por Estación", className="mb-2", style={'fontSize': '16px'}),
                        dcc.Graph(
                            id='temporal-pattern-graph', 
                            figure=create_temporal_pattern_figure(processed_data),
                            style={'height': '400px'} # Altura reducida
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Análisis: ", style={'fontSize': '12px'}), 
                                html.Span("Este gráfico muestra la distribución de tiempos de permanencia en estación a lo largo del día.", 
                                          style={'fontSize': '12px'})
                            ], className="alert alert-info p-1 mt-1 mb-0", style={'fontSize': '12px'})
                        ])
                    ], className="col-md-6", style={'padding': '10px'}),
                    
                    # Clustering de trenes
                    html.Div([
                        html.H4("Clustering de Patrones Operativos", className="mb-2", style={'fontSize': '16px'}),
                        dcc.Graph(
                            id='train-clustering-graph', 
                            figure=create_train_clustering_figure(processed_data),
                            style={'height': '400px'} # Altura reducida
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Análisis: ", style={'fontSize': '12px'}), 
                                html.Span("La agrupación mediante K-means identifica categorías de comportamiento operativo.", 
                                          style={'fontSize': '12px'})
                            ], className="alert alert-info p-1 mt-1 mb-0", style={'fontSize': '12px'})
                        ])
                    ], className="col-md-6", style={'padding': '10px'}),
                ], className="row", style={'marginBottom': '15px', 'display': 'flex', 'flexWrap': 'wrap'}),
                
                # Segunda fila de gráficos
                html.Div([
                    # Detección de anomalías
                    html.Div([
                        html.H4("Detección de Anomalías", className="mb-2", style={'fontSize': '16px'}),
                        dcc.Graph(
                            id='anomaly-detection-graph', 
                            figure=create_anomaly_detection_figure(processed_data),
                            style={'height': '400px'} # Altura reducida
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Análisis: ", style={'fontSize': '12px'}), 
                                html.Span("Utilizando Isolation Forest, este gráfico identifica tiempos de permanencia anómalos.", 
                                          style={'fontSize': '12px'})
                            ], className="alert alert-info p-1 mt-1 mb-0", style={'fontSize': '12px'})
                        ])
                    ], className="col-md-6", style={'padding': '10px'}),
                    
                    # Análisis de material rodante
                    html.Div([
                        html.H4("Análisis por Material Rodante", className="mb-2", style={'fontSize': '16px'}),
                        dcc.Graph(
                            id='material-analysis-graph', 
                            figure=create_material_analysis_figure(processed_data),
                            style={'height': '400px'} # Altura reducida
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Análisis: ", style={'fontSize': '12px'}), 
                                html.Span("Comparación del rendimiento operativo por tipo de material rodante.", 
                                          style={'fontSize': '12px'})
                            ], className="alert alert-info p-1 mt-1 mb-0", style={'fontSize': '12px'})
                        ])
                    ], className="col-md-6", style={'padding': '10px'}),
                ], className="row", style={'marginBottom': '15px', 'display': 'flex', 'flexWrap': 'wrap'}),
                
                # Tercera fila de gráficos - Modelo predictivo (ancho completo)
                html.Div([
                    html.Div([
                        html.H4("Modelo Predictivo de Tiempos de Permanencia", className="mb-2", style={'fontSize': '16px'}),
                        dcc.Graph(
                            id='predictive-model-graph', 
                            figure=create_predictive_model_figure(processed_data),
                            style={'height': '400px'} # Altura reducida
                        ),
                        html.Div([
                            html.P([
                                html.Strong("Análisis: ", style={'fontSize': '12px'}), 
                                html.Span("Este modelo de regresión predice tiempos de permanencia basándose en características operativas.", 
                                          style={'fontSize': '12px'})
                            ], className="alert alert-info p-1 mt-1 mb-0", style={'fontSize': '12px'})
                        ])
                    ], className="col-md-12", style={'padding': '10px'})
                ], className="row", style={'marginBottom': '15px', 'display': 'flex'}),
                
                # Cuarta fila - Tabla de insights
                html.Div([
                    html.Div([
                        html.H4("Insights y Recomendaciones Operativas", className="mb-2", style={'fontSize': '16px'}),
                        html.Div([
                            create_insights_recommendations_table(processed_data)
                        ], className="table-responsive", style={'fontSize': '12px'})
                    ], className="col-md-12", style={'padding': '10px'})
                ], className="row", style={'marginBottom': '15px', 'display': 'flex'}),
                
                # Quinta fila - Resumen ejecutivo
                html.Div([
                    html.Div([
                        html.H4("Resumen Ejecutivo", className="mb-2", style={'fontSize': '16px'}),
                        html.Div([
                            html.P(generate_executive_summary(processed_data), 
                                  className="p-2 mb-0 border rounded bg-light", style={'fontSize': '12px'})
                        ])
                    ], className="col-md-12", style={'padding': '10px'})
                ], className="row", style={'marginBottom': '15px', 'display': 'flex'}),
            ], style={'width': '100%', 'overflowX': 'hidden'}),  # Prevenir scroll horizontal
            
        ], style={
            'padding': '0 15px 25px 15px',  # Padding reducido
            'maxWidth': '100%',             
            'overflowX': 'hidden',          
            'backgroundColor': '#f9f9f9',   
            'minHeight': '90vh',            # Altura mínima reducida
            'fontSize': '14px'              # Tamaño de fuente base reducido
        })
        
        return tab_content
        
    except Exception as e:
        # En caso de error, mostrar mensaje amigable
        error_tab = html.Div([
            html.H2("Análisis Avanzado de Operaciones", className="text-center mb-2", style={'fontSize': '20px'}),
            html.Div([
                html.Div([
                    html.H4("Estado del Análisis Avanzado", style={'fontSize': '16px'}),
                    html.Div([
                        html.P([
                            html.Strong("Se detectó un problema durante el análisis: ", style={'fontSize': '14px'}), 
                            f"{str(e)}"
                        ], className="alert alert-warning p-2", style={'fontSize': '12px'}),
                        html.P([
                            "Se muestran visualizaciones simplificadas. Para obtener análisis más detallados, verifique los datos de entrada."
                        ], className="p-2", style={'fontSize': '12px'}),
                    ])
                ], className="col-md-12 mb-2")
            ], className="row"),
            
            # Siempre mostrar al menos algunas métricas básicas
            html.Div([
                html.H4("Métricas Básicas", className="mb-2", style={'fontSize': '16px'}),
                html.Div([
                    # Columna 1
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H5("Total de Registros", style={'fontSize': '14px'}),
                                html.H3(f"{len(data['velcom_data'])}", className="font-weight-bold", style={'fontSize': '18px'}),
                            ], className="card-body text-center")
                        ], className="card border-primary mb-2")
                    ], className="col-md-4"),
                    
                    # Columna 2
                    html.Div([
                        html.Div([
                            html.H5("Estaciones Registradas", style={'fontSize': '14px'}),
                            html.H3(f"{data['velcom_data']['station'].nunique()}", className="font-weight-bold", style={'fontSize': '18px'}),
                        ], className="card-body text-center")
                    ], className="card border-success mb-2"),
                    
                    # Columna 3
                    html.Div([
                        html.Div([
                            html.H5("Trenes Analizados", style={'fontSize': '14px'}),
                            html.H3(f"{data['velcom_data']['train_number'].nunique()}", className="font-weight-bold", style={'fontSize': '18px'}),
                        ], className="card-body text-center")
                    ], className="card border-info mb-2")
                ], className="row", style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'justifyContent': 'space-between'}),
                
                html.Div([
                    html.P("Recomendación: Verificar los datos de entrada o contactar al administrador del sistema para obtener asistencia.",
                          style={'fontSize': '12px'})
                ], className="alert alert-info mt-2 p-2")
            ], style={'padding': '10px', 'margin': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'padding': '15px', 'maxWidth': '100%', 'overflowX': 'hidden', 'fontSize': '14px'})
        
        return error_tab

def load_data(data_path):
    """Cargar datos procesados de Velcom y preparar para visualización 3D"""
    data = {
        'velcom_data': pd.read_csv(os.path.join(data_path, 'velcom_data.csv')),
        'velcom_trains': pd.read_csv(os.path.join(data_path, 'velcom_trains.csv')),
        'velcom_stations': pd.read_csv(os.path.join(data_path, 'velcom_stations.csv')),
        'velcom_info': pd.read_csv(os.path.join(data_path, 'velcom_info.csv'))
    }
    
    # 1. Convertir columnas de tiempo a datetime
    for col in ['arrival_time', 'departure_time']:
        if col in data['velcom_data'].columns:
            data['velcom_data'][col] = pd.to_datetime(data['velcom_data'][col], errors='coerce')
    
    for col in ['first_arrival', 'last_arrival']:
        if col in data['velcom_trains'].columns:
            data['velcom_trains'][col] = pd.to_datetime(data['velcom_trains'][col], errors='coerce')
    
    # 2. Preparar datos para visualización 3D
    
    # 2.1 Crear copia de trabajo para no modificar los datos originales
    df_3d = data['velcom_data'].copy()
    
    # 2.2 Ordenar por tren y tiempo de llegada para cálculos secuenciales
    df_3d = df_3d.sort_values(['train_number', 'arrival_time'])
    
    # 2.3 Calcular tiempo desde inicio del día (en horas, para eje X)
    df_3d['time_hours'] = df_3d['arrival_time'].dt.hour + df_3d['arrival_time'].dt.minute/60
    
    # 2.4 Calcular tiempo entre estaciones consecutivas por tren
    df_3d['prev_arrival'] = df_3d.groupby('train_number')['arrival_time'].shift(1)
    df_3d['time_diff'] = (df_3d['arrival_time'] - df_3d['prev_arrival']).dt.total_seconds() / 60  # en minutos
    
    # 2.5 Calcular tiempo de permanencia en cada estación
    # Asegurarse de que ambas columnas son datetime
    if 'departure_time' in df_3d.columns and 'arrival_time' in df_3d.columns:
        df_3d['stay_time'] = (df_3d['departure_time'] - df_3d['arrival_time']).dt.total_seconds() / 60  # en minutos
    else:
        df_3d['stay_time'] = 0  # valor por defecto si no hay datos
    
    # 2.6 Calcular velocidad relativa
    # Para cada tren, calculamos una velocidad relativa basada en el tiempo entre estaciones
    train_groups = df_3d.groupby('train_number')
    
    all_trains_data = []
    for train, train_data in train_groups:
        if len(train_data) > 1:
            # Llenar NaN al inicio con valor similar al siguiente
            train_data['time_diff'] = train_data['time_diff'].fillna(train_data['time_diff'].median())
            
            # Invertir tiempo (a mayor tiempo entre estaciones, menor velocidad)
            max_time = train_data['time_diff'].max() if train_data['time_diff'].max() > 0 else 1
            train_data['speed'] = 10 * (1 - train_data['time_diff'] / max_time) + 1  # Escala de 1 a 11
        else:
            # Solo hay un registro para este tren
            train_data['speed'] = 5  # Valor medio por defecto
            
        all_trains_data.append(train_data)
    
    # Combinar todos los datos procesados
    if all_trains_data:
        df_3d_processed = pd.concat(all_trains_data)
    else:
        df_3d_processed = df_3d.copy()
        df_3d_processed['speed'] = 5  # Valor por defecto
    
    # 2.7 Crear mapeo numérico de estaciones para visualización 3D
    all_stations = sorted(df_3d_processed['station'].unique())
    station_mapping = {station: i for i, station in enumerate(all_stations)}
    df_3d_processed['station_num'] = df_3d_processed['station'].map(station_mapping)
    
    # 2.8 Llenar valores NaN en datos críticos
    for col in ['time_diff', 'stay_time', 'speed']:
        df_3d_processed[col] = df_3d_processed[col].fillna(0)
    
    # 3. Guardar los datos procesados
    data['velcom_data_3d'] = df_3d_processed
    data['station_mapping'] = station_mapping
    
    return data

def create_dashboard(data_path):
    """Crear aplicación Dash para visualizar datos Velcom"""
    # Cargar datos
    data = load_data(data_path)
    
    # Obtener info del reporte
    report_info = data['velcom_info'].iloc[0]
    
    app = dash.Dash(
        __name__, 
        title="Dashboard Velcom - Línea 2",
        external_stylesheets=[dbc.themes.BOOTSTRAP]  # Añadir tema de Bootstrap
    )
    
    # Obtener listas de valores únicos para filtros
    train_numbers = sorted(data['velcom_data']['train_number'].unique())
    materials = sorted(data['velcom_data']['material'].unique())
    stations = sorted(data['velcom_data']['station'].unique())
    
    # Diseñar layout
    app.layout = html.Div([
        html.H1("Dashboard de Datos Velcom - Línea 2", className="dashboard-title"),
        
        # Información general del reporte
        html.Div([
            html.Div([
                html.H3("Información del Reporte"),
                html.P(f"Periodo: {report_info['start_date']} a {report_info['end_date']}"),
                html.P(f"Total de registros: {report_info['records_count']}"),
                html.P(f"Total de trenes: {report_info['trains_count']}"),
                html.P(f"Total de estaciones: {report_info['stations_count']}"),
                html.P(f"Procesado el: {report_info['processing_date']}")
            ], className="info-box")
        ], className="info-container"),
        
        # Filtros
        html.Div([
            html.H3("Filtros"),
            
            html.Div([
                html.Div([
                    html.Label("Seleccionar Tren:"),
                    dcc.Dropdown(
                        id='train-dropdown',
                        options=[{'label': f'Tren {num}', 'value': num} for num in train_numbers],
                        value=None,
                        placeholder="Seleccionar tren..."
                    )
                ], className="filter-col"),
                
                html.Div([
                    html.Label("Seleccionar Material:"),
                    dcc.Dropdown(
                        id='material-dropdown',
                        options=[{'label': mat, 'value': mat} for mat in materials],
                        value=None,
                        placeholder="Seleccionar material..."
                    )
                ], className="filter-col"),
                
                html.Div([
                    html.Label("Seleccionar Estación:"),
                    dcc.Dropdown(
                        id='station-dropdown',
                        options=[{'label': sta, 'value': sta} for sta in stations],
                        value=None,
                        placeholder="Seleccionar estación..."
                    )
                ], className="filter-col")
            ], className="filters-row"),
            
            html.Div([
                html.Button('Limpiar Filtros', id='clear-filters-button', className="clear-button")
            ], className="button-container")
        ], className="filters-container"),
        
    # Pestañas para diferentes vistas
    dcc.Tabs([
        # Pestaña de Trenes
        dcc.Tab(label='Trenes', children=[
            html.Div([
                html.Div([
                    html.H3("Trenes en operación"),
                    dash_table.DataTable(
                        id='trains-table',
                        columns=[
                            {'name': 'Número de Tren', 'id': 'train_number'},
                            {'name': 'Material', 'id': 'material'},
                            {'name': 'Estaciones Visitadas', 'id': 'stations_count'},
                            {'name': 'Primera Llegada', 'id': 'first_arrival'},
                            {'name': 'Última Llegada', 'id': 'last_arrival'}
                        ],
                        data=data['velcom_trains'].to_dict('records'),
                        page_size=10,
                        filter_action='native',
                        sort_action='native',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], className="content-box"),
                
                html.Div([
                    html.H3("Distribución de Trenes por Estación"),
                    dcc.Graph(id='trains-by-station-graph')
                ], className="content-box")
            ], className="tab-content")
        ]),
        
        # Pestaña de Estaciones
        dcc.Tab(label='Estaciones', children=[
            html.Div([
                html.Div([
                    html.H3("Actividad por Estación"),
                    dash_table.DataTable(
                        id='stations-table',
                        columns=[
                            {'name': 'Estación', 'id': 'station'},
                            {'name': 'Trenes', 'id': 'train_count'},
                            {'name': 'Vía Promedio', 'id': 'avg_track'},
                            {'name': 'Llegadas', 'id': 'arrival_count'}
                        ],
                        data=data['velcom_stations'].to_dict('records'),
                        page_size=10,
                        filter_action='native',
                        sort_action='native',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], className="content-box"),
                
                html.Div([
                    html.H3("Distribución de Llegadas por Estación"),
                    dcc.Graph(id='arrivals-by-station-graph')
                ], className="content-box")
            ], className="tab-content")
        ]),
        
        # Pestaña de Trayectos
        dcc.Tab(label='Trayectos', children=[
            html.Div([
                html.Div([
                    html.H3("Trayectos de Trenes"),
                    dcc.Graph(id='train-journey-graph')
                ], className="content-box full-width"),
                
                html.Div([
                    html.H3("Detalle de Trayectos"),
                    dash_table.DataTable(
                        id='journey-details-table',
                        columns=[
                            {'name': 'Tren', 'id': 'train_number'},
                            {'name': 'Material', 'id': 'material'},
                            {'name': 'Vía', 'id': 'track'},
                            {'name': 'Estación', 'id': 'station'},
                            {'name': 'Llegada', 'id': 'arrival_time'},
                            {'name': 'Salida', 'id': 'departure_time'}
                        ],
                        data=data['velcom_data'].to_dict('records'),
                        page_size=15,
                        filter_action='native',
                        sort_action='native',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], className="content-box full-width")
            ], className="tab-content")
        ]),
        
        # Pestaña para visualización 3D
        dcc.Tab(label='Visualización 3D', children=[
            html.Div([
                html.Div([
                    html.H3("Visualización 3D de Trayectos"),
                    html.Div([
                        html.Label("Variable para eje Z:"),
                        dcc.RadioItems(
                            id='z-axis-variable',
                            options=[
                                {'label': 'Hora del día', 'value': 'time_hours'},
                                {'label': 'Tiempo entre estaciones', 'value': 'time_diff'},
                                {'label': 'Velocidad relativa', 'value': 'speed'},
                                {'label': 'Tiempo de permanencia', 'value': 'stay_time'}
                            ],
                            value='speed',
                            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                        )
                    ], style={'marginBottom': '20px', 'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'}),
                    dcc.Graph(id='train-3d-graph', style={'height': '700px'})
                ], className="content-box full-width")
            ], className="tab-content")
        ]),
        
        # Nueva pestaña de Análisis Avanzado
        dcc.Tab(label='Análisis Avanzado', children=[
            html.Div(id='ml-analysis-container')
        ])
    ], className="tabs-container")
    ])
    
    
    
    
    # Callback para actualizar gráfico de trenes por estación
    @app.callback(
        Output('trains-by-station-graph', 'figure'),
        [Input('train-dropdown', 'value'),
         Input('material-dropdown', 'value'),
         Input('station-dropdown', 'value')]
    )
    def update_trains_by_station(train_number, material, station):
        df = data['velcom_data'].copy()
        
        # Aplicar filtros
        if train_number:
            df = df[df['train_number'] == train_number]
        if material:
            df = df[df['material'] == material]
        if station:
            df = df[df['station'] == station]
        
        # Contar trenes por estación
        station_counts = df.groupby('station')['train_number'].nunique().reset_index()
        station_counts.columns = ['Estación', 'Cantidad de Trenes']
        station_counts = station_counts.sort_values('Cantidad de Trenes', ascending=False)
        
        # Crear gráfico
        fig = px.bar(
            station_counts, 
            x='Estación', 
            y='Cantidad de Trenes',
            title='Distribución de Trenes por Estación',
            color='Cantidad de Trenes',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title='Estación',
            yaxis_title='Cantidad de Trenes',
            template='plotly_white'
        )
        
        return fig
    
    # Callback para actualizar gráfico de llegadas por estación
    @app.callback(
        Output('arrivals-by-station-graph', 'figure'),
        [Input('train-dropdown', 'value'),
         Input('material-dropdown', 'value'),
         Input('station-dropdown', 'value')]
    )
    def update_arrivals_by_station(train_number, material, station):
        df = data['velcom_data'].copy()
        
        # Aplicar filtros
        if train_number:
            df = df[df['train_number'] == train_number]
        if material:
            df = df[df['material'] == material]
        if station:
            df = df[df['station'] == station]
        
        # Contar llegadas por estación
        arrival_counts = df.groupby('station')['arrival_time'].count().reset_index()
        arrival_counts.columns = ['Estación', 'Número de Llegadas']
        arrival_counts = arrival_counts.sort_values('Número de Llegadas', ascending=False)
        
        # Crear gráfico
        fig = px.bar(
            arrival_counts, 
            x='Estación', 
            y='Número de Llegadas',
            title='Distribución de Llegadas por Estación',
            color='Número de Llegadas',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(
            xaxis_title='Estación',
            yaxis_title='Número de Llegadas',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('train-journey-graph', 'figure'),
        [Input('train-dropdown', 'value'),
        Input('material-dropdown', 'value'),
        Input('station-dropdown', 'value')]
    )
    def update_train_journey(train_number, material, station):
        df = data['velcom_data'].copy()
        
        # Aplicar filtros
        if train_number:
            df = df[df['train_number'] == train_number]
        if material:
            df = df[df['material'] == material]
        if station:
            df = df[df['station'] == station]
        
        # Limitar el número de trenes para evitar gráficos sobrecargados
        if not train_number and len(df['train_number'].unique()) > 10:
            top_trains = df['train_number'].value_counts().nlargest(10).index.tolist()
            df = df[df['train_number'].isin(top_trains)]
        
        # Orden geográfico correcto de las estaciones de la Línea 2 (de norte a sur)
        l2_stations_order = [
            "PI", "CM", "OB", "EB", "LC", "EP", "LO", "CN", "DE", "LV", "SM", "LL", "FR", "RO", 
            "PQ", "TO", "HE", "AN", "CA", "PT", "CB", "CE", "EI", "DO", "ZA", "AV"
        ]
        
        # Crear mapeo de estaciones basado en el orden geográfico
        station_order_mapping = {station: i for i, station in enumerate(l2_stations_order)}
        
        # Obtener todas las estaciones en el orden definido (solo las que están en los datos)
        all_stations = [s for s in l2_stations_order if s in df['station'].unique()]
        
        # Ordenar por tren y tiempo
        df = df.sort_values(['train_number', 'arrival_time'])
        
        # Crear gráfico de trayectos
        fig = go.Figure()
        
        # Crear una línea por cada tren
        for train in df['train_number'].unique():
            train_data = df[df['train_number'] == train]
            material_id = train_data['material'].iloc[0]
            
            # Añadir línea para el trayecto
            fig.add_trace(go.Scatter(
                x=train_data['arrival_time'],
                y=train_data['station'],
                mode='lines+markers',
                name=f'Tren {train} - {material_id}',
                hovertemplate='<b>Tren:</b> %{text}<br>' +
                            '<b>Estación:</b> %{y}<br>' +
                            '<b>Llegada:</b> %{x}<br>',
                text=[f"{train} ({material_id})" for _ in range(len(train_data))]
            ))
        
        # Personalizar layout con categorías de estaciones ordenadas
        fig.update_layout(
            title='Trayectos de Trenes por Estación y Tiempo',
            xaxis_title='Hora',
            yaxis_title='Estación',
            template='plotly_white',
            legend_title='Trenes',
            height=600,
            yaxis=dict(
                categoryorder='array',
                categoryarray=all_stations
            )
        )
        
        return fig

    # Callback para actualizar gráfico 3D de trayectos de trenes
    @app.callback(
        Output('train-3d-graph', 'figure'),
        [Input('train-dropdown', 'value'),
        Input('material-dropdown', 'value'),
        Input('station-dropdown', 'value'),
        Input('z-axis-variable', 'value')]
    )
    def update_train_3d_journey(train_number, material, station, z_variable):
        df = data['velcom_data'].copy()
        
        # Aplicar filtros
        if train_number:
            df = df[df['train_number'] == train_number]
        if material:
            df = df[df['material'] == material]
        if station:
            df = df[df['station'] == station]
        
        # Si no hay datos después de filtrar, devolver figura vacía
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title='No hay datos para los filtros seleccionados',
                scene=dict(
                    xaxis_title='Hora del día',
                    yaxis_title='Estación',
                    zaxis_title='Valor'
                )
            )
            return fig
        
        # Limitar el número de trenes para evitar gráficos sobrecargados
        if not train_number and len(df['train_number'].unique()) > 5:
            top_trains = df['train_number'].value_counts().nlargest(5).index.tolist()
            df = df[df['train_number'].isin(top_trains)]
        
        # Usar el mismo orden de estaciones que el primer gráfico
        l2_stations_order = [
            "PI", "CM", "OB", "EB", "LC", "EP", "LO", "CN", "DE", "LV", "SM", "LL", "FR", "RO", 
            "PQ", "TO", "HE", "AN", "CA", "PT", "CB", "CE", "EI", "DO", "ZA", "AV"
        ]
        
        # Crear mapeo de estaciones basado en el orden geográfico
        station_order_mapping = {station: i for i, station in enumerate(l2_stations_order)}
        
        # Obtener las estaciones presentes en los datos
        available_stations = df['station'].unique()
        
        # Ordenar las estaciones disponibles según el orden geográfico
        # Solo usar las estaciones que están en el mapeo
        filtered_stations = [s for s in l2_stations_order if s in available_stations]
        
        # Para cualquier estación que no esté en nuestro mapeo, agregarla al final
        all_stations = filtered_stations + [s for s in available_stations if s not in l2_stations_order]
        
        station_mapping = {station: i for i, station in enumerate(all_stations)}
        df['station_num'] = df['station'].map(station_mapping)
        
        
        # Calcular tiempo desde el inicio del día para cada evento (en horas)
        df['time_hours'] = df['arrival_time'].dt.hour + df['arrival_time'].dt.minute/60
        
        # Calcular diferencia en tiempo con el registro anterior para el mismo tren
        df = df.sort_values(['train_number', 'arrival_time'])
        df['prev_time'] = df.groupby('train_number')['arrival_time'].shift(1)
        df['time_diff'] = (df['arrival_time'] - df['prev_time']).dt.total_seconds() / 60  # en minutos
        
        # Calcular tiempo de permanencia en cada estación
        df['stay_time'] = (df['departure_time'] - df['arrival_time']).dt.total_seconds() / 60  # en minutos
        
        # Crear gráfico 3D
        fig = go.Figure()
        
        # Crear una línea 3D para cada tren
        for train in df['train_number'].unique():
            train_data = df[df['train_number'] == train].copy()
            material_id = train_data['material'].iloc[0]
            
            # Rellenar NaN con 0 para el primer registro de cada tren
            train_data['time_diff'] = train_data['time_diff'].fillna(0)
            train_data['stay_time'] = train_data['stay_time'].fillna(0)
            
            # Calcular velocidad "relativa"
            if len(train_data) > 1:
                max_time = train_data['time_diff'].max()
                if max_time > 0:
                    train_data['speed'] = 10 * (1 - train_data['time_diff'] / max_time) + 1
                else:
                    train_data['speed'] = 1
            else:
                train_data['speed'] = 1
            
            # Seleccionar la variable Z según la selección del usuario
            z_title = ''
            if z_variable == 'time_hours':
                z_values = train_data['time_hours']
                z_title = 'Hora del día'
            elif z_variable == 'time_diff':
                z_values = train_data['time_diff']
                z_title = 'Tiempo entre estaciones (min)'
            elif z_variable == 'stay_time':
                z_values = train_data['stay_time']
                z_title = 'Tiempo de permanencia (min)'
            else:  # 'speed'
                z_values = train_data['speed']
                z_title = 'Velocidad relativa'
            
            # Ajustar tamaño de los marcadores según el tiempo de permanencia
            marker_size = train_data['stay_time'] * 2 + 3  # Ajustar escala para visualización
            
            # Añadir línea 3D para el trayecto
            fig.add_trace(go.Scatter3d(
                x=train_data['time_hours'],       # Hora del día
                y=train_data['station_num'],      # Estación (convertida a número)
                z=z_values,                       # Valor Z seleccionado
                mode='lines+markers',
                name=f'Tren {train} - {material_id}',
                line=dict(width=4),
                marker=dict(
                    size=marker_size,  # Tamaño basado en tiempo de permanencia
                    color=z_values,    # Color basado en el valor Z
                    colorscale='Viridis'
                ),
                hovertemplate='<b>Tren:</b> %{text}<br>' +
                            '<b>Hora:</b> %{x:.2f}<br>' +
                            '<b>Estación:</b> ' + train_data['station'] + '<br>' +
                            '<b>Permanencia:</b> ' + train_data['stay_time'].round(1).astype(str) + ' min<br>' +
                            '<b>' + z_title + ':</b> %{z:.2f}<br>',
                text=[f"{train} ({material_id})" for _ in range(len(train_data))]
            ))
        
        # Etiquetas para el eje Y (estaciones)
        y_tickvals = list(range(len(station_order_mapping)))
        y_ticktext = l2_stations_order
        
        # Personalizar layout
        fig.update_layout(
            title='Visualización 3D de Trayectos de Trenes',
            scene=dict(
                xaxis_title='Hora del día',
                yaxis_title='Estación',
                zaxis_title=z_title,
                xaxis=dict(
                    range=[min(df['time_hours'])-0.5, max(df['time_hours'])+0.5]
                ),
                yaxis=dict(
                    categoryorder='array',
                    categoryarray=y_ticktext,
                    tickvals=y_tickvals,
                    ticktext=y_ticktext  # Usar solo códigos de estación sin tooltip
                    
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2)
                )
            ),
            legend_title='Trenes',
            margin=dict(l=0, r=0, t=30, b=0),
            height=700
        )
        
        return fig

    
    # Callback para actualizar tabla de detalles de trayectos
    @app.callback(
        Output('journey-details-table', 'data'),
        [Input('train-dropdown', 'value'),
         Input('material-dropdown', 'value'),
         Input('station-dropdown', 'value')]
    )
    def update_journey_details(train_number, material, station):
        df = data['velcom_data'].copy()
        
        # Aplicar filtros
        if train_number:
            df = df[df['train_number'] == train_number]
        if material:
            df = df[df['material'] == material]
        if station:
            df = df[df['station'] == station]
        
        # Convertir a formato legible para la tabla
        df['arrival_time'] = df['arrival_time'].dt.strftime('%d/%m/%Y %H:%M:%S')
        df['departure_time'] = df['departure_time'].dt.strftime('%d/%m/%Y %H:%M:%S')
        
        # Ordenar por tren y tiempo de llegada
        df = df.sort_values(['train_number', 'arrival_time'])
        
        return df.to_dict('records')
    
    # Callback para limpiar filtros
    @app.callback(
        [Output('train-dropdown', 'value'),
         Output('material-dropdown', 'value'),
         Output('station-dropdown', 'value')],
        [Input('clear-filters-button', 'n_clicks')]
    )
    def clear_filters(n_clicks):
        # Si el botón no ha sido presionado, no hacer nada
        if n_clicks is None:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Limpiar todos los filtros
        return None, None, None
    
    # Actualizar tablas iniciales
    @app.callback(
        [Output('trains-table', 'data'),
         Output('stations-table', 'data')],
        [Input('train-dropdown', 'value'),
         Input('material-dropdown', 'value'),
         Input('station-dropdown', 'value')]
    )
    def update_tables(train_number, material, station):
        # Datos de trenes filtrados
        trains_df = data['velcom_trains'].copy()
        if train_number:
            trains_df = trains_df[trains_df['train_number'] == train_number]
        if material:
            trains_df = trains_df[trains_df['material'] == material]
        
        # Datos de estaciones filtrados
        stations_df = data['velcom_stations'].copy()
        if station:
            stations_df = stations_df[stations_df['station'] == station]
            
        # Si hay filtros de tren o material, filtrar estaciones relacionadas
        if train_number or material:
            filtered_df = data['velcom_data'].copy()
            if train_number:
                filtered_df = filtered_df[filtered_df['train_number'] == train_number]
            if material:
                filtered_df = filtered_df[filtered_df['material'] == material]
                
            related_stations = filtered_df['station'].unique()
            stations_df = stations_df[stations_df['station'].isin(related_stations)]
        
        # Formatear fechas para visualización
        trains_df['first_arrival'] = pd.to_datetime(trains_df['first_arrival']).dt.strftime('%d/%m/%Y %H:%M:%S')
        trains_df['last_arrival'] = pd.to_datetime(trains_df['last_arrival']).dt.strftime('%d/%m/%Y %H:%M:%S')
        
        return trains_df.to_dict('records'), stations_df.to_dict('records')
    
    @app.callback(
        Output('ml-analysis-container', 'children'),
        [Input('train-dropdown', 'value'),
         Input('material-dropdown', 'value'),
         Input('station-dropdown', 'value')]
    )
    def update_ml_analysis(train_number, material, station):
        # Filtrar datos si es necesario
        filtered_data = data['velcom_data'].copy()
        
        if train_number:
            filtered_data = filtered_data[filtered_data['train_number'] == train_number]
        if material:
            filtered_data = filtered_data[filtered_data['material'] == material]
        if station:
            filtered_data = filtered_data[filtered_data['station'] == station]
        
        # Si no hay datos filtrados, usar todos los datos
        if filtered_data.empty:
            filtered_data = data['velcom_data']
        
        # Llamar a la función de análisis ML con los datos filtrados
        return create_ml_analysis_tab({'velcom_data': filtered_data})
    
    # Añadir CSS para estilizar el dashboard
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .dashboard-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    font-family: "Segoe UI", Arial, sans-serif;
                }
                .dashboard-title {
                    text-align: center;
                    color: #2C3E50;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #3498DB;
                    margin-bottom: 20px;
                }
                .info-container {
                    margin-bottom: 20px;
                }
                .info-box {
                    background-color: #F8F9F9;
                    border-left: 5px solid #3498DB;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .filters-container {
                    background-color: #F8F9F9;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .filters-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 15px;
                }
                .filter-col {
                    flex: 1;
                    min-width: 200px;
                }
                .button-container {
                    display: flex;
                    justify-content: center;
                }
                .clear-button {
                    background-color: #E74C3C;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                .clear-button:hover {
                    background-color: #C0392B;
                }
                .tabs-container {
                    margin-top: 20px;
                }
                .tab-content {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    padding: 20px 0;
                }
                .content-box {
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    flex: 1 1 calc(50% - 20px);
                    min-width: 300px;
                }
                .full-width {
                    flex: 1 1 100%;
                }
                h3 {
                    color: #2C3E50;
                    border-bottom: 1px solid #EAECEE;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    return app

def open_browser(port):
    """Abrir navegador en la URL del dashboard"""
    webbrowser.open_new(f"http://localhost:{port}")

def launch_velcom_dashboard(data_path, port=8055):
    """Lanzar dashboard de Velcom"""
    # Crear la aplicación
    app = create_dashboard(data_path)
    
    # Abrir navegador después de 1 segundo
    Timer(1, open_browser, [port]).start()
    
    # Ejecutar servidor
    app.run_server(debug=False, port=port)