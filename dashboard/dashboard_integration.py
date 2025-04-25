# dashboard/dashboard_integration.py
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
from dashboard.dashboard_generator import DashboardGenerator

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import threading
import queue
import webbrowser
import logging
from tkinter import messagebox

class DashboardIntegration:
    def __init__(self, output_folder, parent_window=None):
        self.output_folder = output_folder
        self.parent_window = parent_window
        self.dashboards = {}  # Almacenar referencias a los dashboards activos
        self.ports = {}  # Seguimiento de puertos utilizados
        self.next_port = 8050  # Puerto inicial
        self.dashboard_threads = {}  # Almacenar hilos de dashboards
        self.stop_events = {}  # Eventos para detener hilos
        
    def launch_dashboard(self, line, analysis_type):
        try:
            # Verificar si el dashboard ya está en ejecución
            dashboard_key = f"{line}_{analysis_type}"
            
            # Detener dashboard existente si está corriendo
            if dashboard_key in self.dashboards:
                self.stop_dashboard(line, analysis_type)
            
            # Crear dashboard
            from dashboard.dashboard_generator import DashboardGenerator
            dashboard = DashboardGenerator(self.output_folder, line, analysis_type)
            
            # Cargar datos
            if not dashboard.load_data():
                messagebox.showerror("Error", "No se pudieron cargar los datos")
                return False
            
            # Crear dashboard
            if not dashboard.create_dashboard():
                messagebox.showerror("Error", "No se pudo crear el dashboard")
                return False
            
            # Configurar puerto
            if dashboard_key not in self.ports:
                self.ports[dashboard_key] = self.next_port
                self.next_port += 1
            
            # Configurar evento de parada
            stop_event = threading.Event()
            self.stop_events[dashboard_key] = stop_event
            
            # Función para ejecutar dashboard
            def run_dashboard_thread():
                try:
                    dashboard.port = self.ports[dashboard_key]
                    dashboard.run_dashboard()
                except Exception as e:
                    logging.error(f"Error en dashboard {dashboard_key}: {e}")
                finally:
                    stop_event.set()
            
            # Crear y iniciar hilo
            thread = threading.Thread(
                target=run_dashboard_thread, 
                daemon=True
            )
            thread.start()
            self.dashboard_threads[dashboard_key] = thread
            
            # Almacenar referencias
            self.dashboards[dashboard_key] = dashboard
            
            # Abrir en navegador
            import time
            time.sleep(1)  # Pequeño retraso para asegurar que el servidor esté arriba
            webbrowser.open(f"http://localhost:{dashboard.port}")
            
            return True
        
        except Exception as e:
            logging.error(f"Error al lanzar dashboard: {e}")
            messagebox.showerror("Error", str(e))
            return False
    
    def stop_dashboard(self, line, analysis_type):
        """Detener un dashboard específico"""
        dashboard_key = f"{line}_{analysis_type}"
        
        try:
            # Detener el dashboard si existe
            if dashboard_key in self.dashboards:
                dashboard = self.dashboards[dashboard_key]
                
                # Intentar detener el servidor
                if hasattr(dashboard, 'stop_dashboard'):
                    dashboard.stop_dashboard()
                
                # Detener el hilo
                if dashboard_key in self.stop_events:
                    self.stop_events[dashboard_key].set()
                
                # Eliminar referencias
                del self.dashboards[dashboard_key]
                
                if dashboard_key in self.dashboard_threads:
                    del self.dashboard_threads[dashboard_key]
                
                if dashboard_key in self.ports:
                    del self.ports[dashboard_key]
                
                if dashboard_key in self.stop_events:
                    del self.stop_events[dashboard_key]
        
        except Exception as e:
            logging.error(f"Error al detener dashboard: {e}")
    
    def stop_all_dashboards(self):
        """Detener todos los dashboards activos"""
        dashboard_keys = list(self.dashboards.keys())
        for key in dashboard_keys:
            line, analysis_type = key.split('_')
            self.stop_dashboard(line, analysis_type)
    
    def _launch_dashboard_thread(self, line, analysis_type, port):
        """Método interno para lanzar el dashboard en un hilo separado"""
        try:
            # Mostrar diálogo de progreso
            if self.parent_window:
                progress_dialog = self._create_progress_dialog(f"Generando dashboard para {line} {analysis_type}...")
            
            # Crear el dashboard
            dashboard = DashboardGenerator(self.output_folder, line, analysis_type, port)
            
            # Cargar datos
            if self.parent_window:
                progress_dialog.update_status("Cargando datos...")
            
            data_loaded = dashboard.load_data()
            if not data_loaded:
                if self.parent_window:
                    progress_dialog.destroy()
                    messagebox.showerror("Error", f"No se pudieron cargar los datos para {line} {analysis_type}")
                return
            
            # Crear dashboard
            if self.parent_window:
                progress_dialog.update_status("Generando visualizaciones y análisis...")
            
            dashboard_created = dashboard.create_dashboard()
            if not dashboard_created:
                if self.parent_window:
                    progress_dialog.destroy()
                    messagebox.showerror("Error", f"No se pudo crear el dashboard para {line} {analysis_type}")
                return
            
            # Lanzar dashboard
            if self.parent_window:
                progress_dialog.update_status("Iniciando servidor web...")
            
            dashboard_launched = dashboard.run_dashboard()
            
            # Guardar referencia al dashboard
            dashboard_key = f"{line}_{analysis_type}"
            self.dashboards[dashboard_key] = dashboard
            
            # Cerrar diálogo de progreso
            if self.parent_window:
                progress_dialog.destroy()
                
                if dashboard_launched:
                    messagebox.showinfo("Dashboard", f"Dashboard para {line} {analysis_type} iniciado correctamente.\n\nSe ha abierto en su navegador web.")
                else:
                    messagebox.showerror("Error", f"No se pudo iniciar el servidor para el dashboard de {line} {analysis_type}")
            
        except Exception as e:
            logger.error(f"Error en hilo de dashboard: {str(e)}")
            if self.parent_window:
                try:
                    progress_dialog.destroy()
                except:
                    pass
                messagebox.showerror("Error", f"Error al generar el dashboard: {str(e)}")
    
    def _create_progress_dialog(self, initial_message):
        """Crear diálogo de progreso"""
        progress_dialog = ProgressDialog(self.parent_window, initial_message)
        return progress_dialog
    
    def stop_all_dashboards(self):
        """Detener todos los dashboards activos"""
        for key, dashboard in self.dashboards.items():
            if dashboard.running:
                dashboard.stop_dashboard()
        
        self.dashboards = {}
        logger.info("Todos los dashboards detenidos")
        return True


class ProgressDialog:
    """Diálogo de progreso simple"""
    
    def __init__(self, parent, initial_message):
        self.top = tk.Toplevel(parent)
        self.top.title("Generando Dashboard")
        self.top.geometry("400x150")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)
        
        # Centrar en la pantalla
        self.center_window()
        
        # Configurar UI
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(0, weight=1)
        self.top.rowconfigure(1, weight=1)
        
        # Mensaje
        self.message_var = tk.StringVar(value=initial_message)
        message_label = ttk.Label(self.top, textvariable=self.message_var, wraplength=380, justify='center')
        message_label.grid(row=0, column=0, padx=20, pady=10)
        
        # Barra de progreso indeterminada
        self.progress_bar = ttk.Progressbar(self.top, mode='indeterminate', length=360)
        self.progress_bar.grid(row=1, column=0, padx=20, pady=20)
        self.progress_bar.start(10)
        
        # Actualizar la UI
        self.top.update()
    
    def update_status(self, message):
        """Actualizar mensaje de estado"""
        self.message_var.set(message)
        self.top.update()
    
    def center_window(self):
        """Centrar ventana en la pantalla"""
        self.top.update_idletasks()
        width = self.top.winfo_width()
        height = self.top.winfo_height()
        x = (self.top.winfo_screenwidth() // 2) - (width // 2)
        y = (self.top.winfo_screenheight() // 2) - (height // 2)
        self.top.geometry(f'{width}x{height}+{x}+{y}')
    
    def destroy(self):
        """Cerrar el diálogo"""
        self.top.grab_release()
        self.top.destroy()