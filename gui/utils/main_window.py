# gui/main_window.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from datetime import datetime
from ttkthemes import ThemedTk

from processors.cdv_processor_l1 import CDVProcessorL1
from processors.adv_processor_l1 import ADVProcessorL1
from processors.cdv_processor_l2 import CDVProcessorL2
from processors.adv_processor_l2 import ADVProcessorL2
from processors.cdv_processor_l4 import CDVProcessorL4
from processors.adv_processor_l4 import ADVProcessorL4
from processors.cdv_processor_l4a import CDVProcessorL4A
from processors.adv_processor_l4a import ADVProcessorL4A
from processors.cdv_processor_l5 import CDVProcessorL5
from processors.adv_processor_l5 import ADVProcessorL5
from gui.line_tabs import LineTab
from gui.utils.config import Config

class MetroAnalyzerApp:
    """Aplicación principal para análisis de datos del Metro"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Metro de Santiago - Analizador SCADA")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        # Cargar configuración
        self.config = Config()
        
        # Variables para seguimiento de progreso
        self.message_queue = queue.Queue()
        self.processing_threads = {}  # Diccionario para almacenar múltiples hilos
        
        # Procesadores para cada línea y tipo de análisis
        self.processors = {
            "L1": {
                "CDV": CDVProcessorL1,
                "ADV": ADVProcessorL1
            },
            "L2": {
                "CDV": CDVProcessorL2,
                "ADV": ADVProcessorL2
            },
            "L4": {
                "CDV": CDVProcessorL4,
                "ADV": ADVProcessorL4
            },
            "L4A": {
                "CDV": CDVProcessorL4A,
                "ADV": ADVProcessorL4A
            },
            "L5": {
                "CDV": CDVProcessorL5,
                "ADV": ADVProcessorL5
            }
        }
        
        # Crear interfaz
        self.create_widgets()
        
        # Configurar actualización de mensajes
        self.root.after(100, self.check_message_queue)
    
    def create_widgets(self):
        """Crear los widgets de la interfaz de usuario"""
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear pestañas para cada línea
        self.tabs = {
            "L1": LineTab(self.notebook, "Línea 1", self),
            "L2": LineTab(self.notebook, "Línea 2", self),  # Habilitada ahora
            "L4": LineTab(self.notebook, "Línea 4", self),
            "L4A": LineTab(self.notebook, "Línea 4A", self),
            "L5": LineTab(self.notebook, "Línea 5", self)
        }
        
        # Añadir pestañas al notebook
        for line, tab in self.tabs.items():
            self.notebook.add(tab.frame, text=tab.title)
            
        # AGREGAR AQUÍ: Gestionar enlaces de eventos cuando se cambia de pestaña
        def handle_tab_changed(event):
            # Obtener la pestaña seleccionada
            tab_id = event.widget.select()
            tab_name = event.widget.tab(tab_id, "text")
            
            # Desactivar eventos de rueda en todas las pestañas
            for line, tab in self.tabs.items():
                if hasattr(tab, 'unbind_events'):
                    tab.unbind_events()
            
            # Activar eventos de rueda solo en la pestaña seleccionada
            for line, tab in self.tabs.items():
                if tab.title == tab_name and hasattr(tab, 'bind_events'):
                    tab.bind_events()

        # Enlazar el evento de cambio de pestaña
        self.notebook.bind("<<NotebookTabChanged>>", handle_tab_changed)
    
    def check_message_queue(self):
        """Verificar mensajes en la cola y actualizar la UI"""
        try:
            while not self.message_queue.empty():
                line, analysis_type, progress, message = self.message_queue.get_nowait()
                
                # Actualizar UI en la pestaña correspondiente
                if line in self.tabs:
                    self.tabs[line].update_progress(analysis_type, progress, message)
                
                # Marcar mensaje como procesado
                self.message_queue.task_done()
        except Exception as e:
            print(f"Error al procesar mensajes: {str(e)}")
        
        # Programar próxima verificación
        self.root.after(100, self.check_message_queue)
    
    # Actualización del método start_processing en gui/utils/main_window.py

    def start_processing(self, line, analysis_type, source_path, dest_path, parameters=None):
        """Iniciar procesamiento para una línea y tipo de análisis específico"""
        # Verificar si existe el procesador para la combinación
        if line not in self.processors or analysis_type not in self.processors[line]:
            messagebox.showerror("Error", f"No se encontró un procesador para Línea {line}, tipo {analysis_type}")
            return False
        
        # Crear una nueva instancia del procesador para permitir ejecuciones simultáneas
        processor_class = self.processors[line][analysis_type]
        processor = processor_class()
        
        # Configurar rutas
        processor.set_paths(source_path, dest_path)
        
        # Configurar parámetros adicionales si existen
        if parameters:
            for param, value in parameters.items():
                if param == "data_type" and hasattr(processor, "set_data_type"):
                    # Método específico para establecer tipo de datos
                    processor.set_data_type(value)
                elif hasattr(processor, param):
                    setattr(processor, param, value)
        
        # Iniciar procesamiento en un hilo separado
        thread_key = f"{line}_{analysis_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        processing_thread = threading.Thread(
            target=self.run_processing,
            args=(line, analysis_type, processor, thread_key)
        )
        processing_thread.daemon = True
        
        # Guardar referencia al hilo
        self.processing_threads[thread_key] = processing_thread
        
        # Iniciar el hilo
        processing_thread.start()
        
        return True
    
    def run_processing(self, line, analysis_type, processor, thread_key):
        """Ejecutar procesamiento en segundo plano"""
        try:
            # Función anónima para retransmitir actualizaciones de progreso
            progress_callback = lambda progress, message: self.message_queue.put((line, analysis_type, progress, message))
            
            # Ejecutar procesamiento
            success = processor.process_data(progress_callback)
            
            if success:
                progress_callback(100, f"Procesamiento de {analysis_type} completado con éxito")
            else:
                progress_callback(0, "Error en el procesamiento")
                
            # Eliminar la referencia al hilo cuando termina
            if thread_key in self.processing_threads:
                del self.processing_threads[thread_key]
                
        except Exception as e:
            self.message_queue.put((line, analysis_type, 0, f"Error: {str(e)}"))
            
            # Eliminar la referencia al hilo en caso de error
            if thread_key in self.processing_threads:
                del self.processing_threads[thread_key]