# gui/line_tabs.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from datetime import datetime
from dashboard.dashboard_integration import DashboardIntegration

class LineTab:
    """Clase para gestionar las pestañas de cada línea"""
    
    def __init__(self, notebook, title, parent_app, enabled=True):
        self.notebook = notebook
        self.title = title
        self.parent_app = parent_app
        
        # Modificación para habilitar solamente habilitar Línea 2 y resto de pestañas se encuentran en desarrollo
        #self.enabled = title == "Línea 2" if enabled else False
        self.enabled = True
        
        # Variables para rutas y configuración defecto mismo codigo anterior pra lib y ocup
        self.source_path_var = tk.StringVar()
        self.dest_path_var = tk.StringVar()
        self.f_oc_1_var = tk.StringVar(value="0.1")
        self.f_lb_2_var = tk.StringVar(value="0.05")
        self.analysis_type_var = tk.StringVar(value="CDV")
        
        # Variable para tipo de datos (Sacem o SCADA) - SCADA en desarrollo (consultar 17-abr)
        self.data_type_var = tk.StringVar(value="Sacem")
        
        # Variables para seguimiento de progreso
        self.progress_var_cdv = tk.DoubleVar()
        self.progress_var_adv = tk.DoubleVar()
        self.status_var_cdv = tk.StringVar(value="Listo para procesar CDV")
        self.status_var_adv = tk.StringVar(value="Listo para procesar ADV")
        
        # Estado de procesamiento
        self.cdv_processing_complete = False
        self.adv_processing_complete = False
        
        # Integración del dashboard
        self.dashboard_integration = None
        
        # Crear el frame principal para la pestaña
        self.frame = ttk.Frame(notebook)
        
        if self.enabled:
            self.create_widgets()
        else:
            self.create_disabled_widgets()
    
    def create_widgets(self):
        """Crear los widgets de la interfaz para una línea habilitada con scrollbar"""
        # Crear un canvas con scrollbar para permitir desplazamiento
        self.canvas = tk.Canvas(self.frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configurar el scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        # Crear ventana dentro del canvas que se expande al ancho completo
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configurar el canvas para que se expanda a todo el ancho disponible
        def configure_canvas(event):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        self.canvas.bind("<Configure>", configure_canvas)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Empaquetar canvas y scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Vincular evento de rueda de ratón para scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Para Linux/Unix también añadimos eventos de botón
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        # Marco principal (ahora dentro del scrollable_frame)
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sección de selección de análisis
        analysis_frame = ttk.LabelFrame(main_frame, text="Tipo de Análisis", padding="10")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Radio buttons para seleccionar CDV o ADV
        ttk.Radiobutton(analysis_frame, text="Circuitos de Vía (CDV)", variable=self.analysis_type_var, 
                    value="CDV", command=self.toggle_analysis_type).grid(row=0, column=0, padx=20, pady=5, sticky=tk.W)
        ttk.Radiobutton(analysis_frame, text="Agujas (ADV)", variable=self.analysis_type_var, 
                    value="ADV", command=self.toggle_analysis_type).grid(row=0, column=1, padx=20, pady=5, sticky=tk.W)
        
        # Sección de selección de tipo de datos (solo para Línea 2)
        if self.title == "Línea 2":
            data_type_frame = ttk.LabelFrame(main_frame, text="Tipo de Datos", padding="10")
            data_type_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Radiobutton(data_type_frame, text="Datos Sacem", variable=self.data_type_var, 
                        value="Sacem").grid(row=0, column=0, padx=20, pady=5, sticky=tk.W)
            ttk.Radiobutton(data_type_frame, text="Datos SCADA (En desarrollo)", variable=self.data_type_var, 
                        value="SCADA", state=tk.DISABLED).grid(row=0, column=1, padx=20, pady=5, sticky=tk.W)
        
        # Sección de selección de carpetas
        folder_frame = ttk.LabelFrame(main_frame, text="Rutas de archivos", padding="10")
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ruta de origen
        ttk.Label(folder_frame, text="Carpeta de origen:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.source_path_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="Examinar...", command=self.browse_source).grid(row=0, column=2, padx=5, pady=5)
        
        # Ruta de destino
        ttk.Label(folder_frame, text="Carpeta de destino:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(folder_frame, textvariable=self.dest_path_var, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="Examinar...", command=self.browse_dest).grid(row=1, column=2, padx=5, pady=5)
        
        # Marco para configuración de parámetros CDV
        self.cdv_config_frame = ttk.LabelFrame(main_frame, text="Configuración CDV", padding="10")
        self.cdv_config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Factor de umbral para ocupación
        ttk.Label(self.cdv_config_frame, text="Factor umbral ocupación (f_oc_1):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.cdv_config_frame, textvariable=self.f_oc_1_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.cdv_config_frame, 
                text="Valor entre 0 y 1. Menor valor = más sensible en detección de fallos de ocupación").grid(
            row=0, column=2, sticky=tk.W, padx=10)
        
        # Factor de umbral para liberación
        ttk.Label(self.cdv_config_frame, text="Factor umbral liberación (f_lb_2):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.cdv_config_frame, textvariable=self.f_lb_2_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.cdv_config_frame, 
                text="Valor entre 0 y 1. Menor valor = más sensible en detección de fallos de liberación").grid(
            row=1, column=2, sticky=tk.W, padx=10)
        
        # Después de la sección de configuración CDV, añadir sección para Velcom (solo para Línea 2)
        if self.title == "Línea 2":
            self.velcom_frame = ttk.LabelFrame(main_frame, text="Procesar Datos Velcom", padding="10")
            self.velcom_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Variables para Velcom
            self.velcom_file_var = tk.StringVar()
            self.velcom_output_var = tk.StringVar()
            
            # Selector de archivo Velcom
            ttk.Label(self.velcom_frame, text="Archivo Velcom (.dat):").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Entry(self.velcom_frame, textvariable=self.velcom_file_var, width=60).grid(row=0, column=1, padx=5, pady=5)
            ttk.Button(self.velcom_frame, text="Examinar...", command=self.browse_velcom_file).grid(row=0, column=2, padx=5, pady=5)
            
            # Carpeta de salida para resultados Velcom
            ttk.Label(self.velcom_frame, text="Carpeta de resultados:").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Entry(self.velcom_frame, textvariable=self.velcom_output_var, width=60).grid(row=1, column=1, padx=5, pady=5)
            ttk.Button(self.velcom_frame, text="Examinar...", command=self.browse_velcom_output).grid(row=1, column=2, padx=5, pady=5)
            
            # Barra de progreso y botón para procesar
            ttk.Label(self.velcom_frame, text="Progreso:").grid(row=2, column=0, sticky=tk.W, pady=5)
            self.velcom_progress_var = tk.DoubleVar()
            self.velcom_progress_bar = ttk.Progressbar(self.velcom_frame, variable=self.velcom_progress_var, length=550, mode="determinate")
            self.velcom_progress_bar.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
            
            # Botón para procesar
            self.process_velcom_button = ttk.Button(self.velcom_frame, text="Procesar Velcom", command=self.process_velcom)
            self.process_velcom_button.grid(row=2, column=2, padx=5, pady=5)
            
            # Status de procesamiento
            self.velcom_status_var = tk.StringVar(value="Listo para procesar Velcom")
            status_label_velcom = ttk.Label(self.velcom_frame, textvariable=self.velcom_status_var, wraplength=600)
            status_label_velcom.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            # Botón para visualizar dashboard
            self.view_velcom_button = ttk.Button(
                self.velcom_frame, 
                text="Visualizar Dashboard Velcom", 
                command=self.view_velcom_dashboard, 
                state=tk.DISABLED
            )
            self.view_velcom_button.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        
        # Sección de progreso y estado
        self.progress_frame = ttk.LabelFrame(main_frame, text="Progreso del Procesamiento", padding="10")
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progreso y estado para CDV
        ttk.Label(self.progress_frame, text="CDV:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.progress_bar_cdv = ttk.Progressbar(self.progress_frame, variable=self.progress_var_cdv, length=550, mode="determinate")
        self.progress_bar_cdv.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Botón para visualizar resultados CDV (inicialmente deshabilitado)
        self.view_cdv_button = ttk.Button(self.progress_frame, text="Visualizar Resultados CDV", command=lambda: self.view_results("CDV"), state=tk.DISABLED)
        self.view_cdv_button.grid(row=0, column=2, padx=5, pady=5)
        
        status_label_cdv = ttk.Label(self.progress_frame, textvariable=self.status_var_cdv, wraplength=600)
        status_label_cdv.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Progreso y estado para ADV
        ttk.Label(self.progress_frame, text="ADV:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.progress_bar_adv = ttk.Progressbar(self.progress_frame, variable=self.progress_var_adv, length=550, mode="determinate")
        self.progress_bar_adv.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Botón para visualizar resultados ADV (inicialmente deshabilitado)
        self.view_adv_button = ttk.Button(self.progress_frame, text="Visualizar Resultados ADV", command=lambda: self.view_results("ADV"), state=tk.DISABLED)
        self.view_adv_button.grid(row=2, column=2, padx=5, pady=5)
        
        status_label_adv = ttk.Label(self.progress_frame, textvariable=self.status_var_adv, wraplength=600)
        status_label_adv.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Área de log
        log_frame = ttk.LabelFrame(main_frame, text="Registro de actividad", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Área de texto con scroll para el log
        self.log_text = tk.Text(log_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        # Usar grid() para mejor control de posición
        ttk.Button(button_frame, text="Analizar CDV", command=lambda: self.start_processing("CDV")).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="Analizar ADV", command=lambda: self.start_processing("ADV")).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="Analizar Ambos", command=self.start_both_processing).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="Limpiar log", command=self.clear_log).grid(row=0, column=0, padx=5, pady=5)
        
    def _on_mousewheel(self, event):
        """Método para permitir el scroll con la rueda del ratón"""
        # Diferentes plataformas tienen diferentes eventos de rueda
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:  # Windows/Mac
            if hasattr(event, 'delta'):  # Windows
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:  # Mac
                self.canvas.yview_scroll(int(-1*event.delta), "units")
    def create_disabled_widgets(self):
        """Crear widgets para pestañas deshabilitadas"""
        ttk.Label(self.frame, text=f"Análisis de {self.title} en desarrollo...", font=("Helvetica", 14)).pack(pady=100)
        ttk.Button(self.frame, text="Volver a Línea 2", command=lambda: self.notebook.select(1)).pack()
    
    def toggle_analysis_type(self):
        """Cambiar configuración según el tipo de análisis seleccionado"""
        analysis_type = self.analysis_type_var.get()
        
        if analysis_type == "CDV":
            self.cdv_config_frame.pack(fill=tk.X, padx=5, pady=5)
            self.log(f"Tipo de análisis seleccionado: Circuitos de Vía (CDV)")
        else:
            self.cdv_config_frame.pack_forget()
            self.log(f"Tipo de análisis seleccionado: Agujas (ADV)")
    
    def browse_source(self):
        """Abrir diálogo para seleccionar carpeta de origen"""
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta de origen")
        if folder_path:
            self.source_path_var.set(folder_path)
            self.log(f"Carpeta de origen seleccionada: {folder_path}")
    
    def browse_dest(self):
        """Abrir diálogo para seleccionar carpeta de destino"""
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta de destino")
        if folder_path:
            self.dest_path_var.set(folder_path)
            self.log(f"Carpeta de destino seleccionada: {folder_path}")
    
    def browse_velcom_file(self):
        """Seleccionar archivo Velcom"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo Velcom",
            filetypes=[("Archivos DAT", "*.dat"), ("Archivos TXT", "*.txt"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            self.velcom_file_var.set(file_path)
            self.log(f"Archivo Velcom seleccionado: {file_path}")

    def browse_velcom_output(self):
        """Seleccionar carpeta de salida para resultados Velcom"""
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta para resultados Velcom")
        if folder_path:
            self.velcom_output_var.set(folder_path)
            self.log(f"Carpeta de resultados Velcom seleccionada: {folder_path}")
    
    def log(self, message):
        """Añadir mensaje al área de log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def clear_log(self):
        """Limpiar el área de log"""
        self.log_text.delete(1.0, tk.END)
        self.log("Log limpiado")
    
    def start_processing(self, analysis_type):
        """Iniciar procesamiento de datos para un tipo específico"""
        # Verificar que se hayan seleccionado las carpetas
        source_path = self.source_path_var.get()
        dest_path = self.dest_path_var.get()
        
        if not source_path or not os.path.exists(source_path):
            messagebox.showerror("Error", "Seleccione una carpeta de origen válida")
            return
        
        if not dest_path or not os.path.exists(dest_path):
            messagebox.showerror("Error", "Seleccione una carpeta de destino válida")
            return
        
        # Obtener datos para procesamiento
        line = self.title.replace("Línea ", "L")
        
        # Obtener tipo de datos para Línea 2
        data_type = "Sacem"  # Valor por defecto
        if hasattr(self, 'data_type_var'):
            data_type = self.data_type_var.get()
            
            # Verificar si se seleccionó SCADA para L2 (aún no implementado)
            if data_type == "SCADA" and line == "L2":
                messagebox.showinfo("Información", "El análisis con datos SCADA para Línea 2 está en desarrollo y no disponible en esta versión.")
                self.log("El análisis con datos SCADA para Línea 2 está en desarrollo")
                return
        
        # Verificar umbrales para CDV
        parameters = {}
        if analysis_type == "CDV":
            try:
                f_oc_1 = float(self.f_oc_1_var.get())
                f_lb_2 = float(self.f_lb_2_var.get())
                
                if not (0 < f_oc_1 <= 1) or not (0 < f_lb_2 <= 1):
                    messagebox.showerror("Error", "Los factores de umbral deben estar entre 0 y 1")
                    return
                
                parameters = {
                    'f_oc_1': f_oc_1,
                    'f_lb_2': f_lb_2
                }
            except ValueError:
                messagebox.showerror("Error", "Los factores de umbral deben ser valores numéricos")
                return
        
        # Añadir tipo de datos a los parámetros
        parameters['data_type'] = data_type
        
        # Registrar en el log el tipo de datos seleccionado
        if line == "L2":
            self.log(f"Tipo de datos seleccionado: {data_type}")
        
        # Reiniciar la barra de progreso correspondiente
        if analysis_type == "CDV":
            self.progress_var_cdv.set(0)
            self.status_var_cdv.set(f"Iniciando procesamiento para {analysis_type}...")
            self.cdv_processing_complete = False
            self.view_cdv_button.config(state=tk.DISABLED)
        else:
            self.progress_var_adv.set(0)
            self.status_var_adv.set(f"Iniciando procesamiento para {analysis_type}...")
            self.adv_processing_complete = False
            self.view_adv_button.config(state=tk.DISABLED)
        
        # Iniciar procesamiento
        success = self.parent_app.start_processing(line, analysis_type, source_path, dest_path, parameters)
        
        if not success:
            self.log(f"Error al iniciar el procesamiento de {analysis_type}")
    
    def start_both_processing(self):
        """Iniciar procesamiento tanto para CDV como para ADV"""
        # Verificar que se hayan seleccionado las carpetas
        source_path = self.source_path_var.get()
        dest_path = self.dest_path_var.get()
        
        if not source_path or not os.path.exists(source_path):
            messagebox.showerror("Error", "Seleccione una carpeta de origen válida")
            return
        
        if not dest_path or not os.path.exists(dest_path):
            messagebox.showerror("Error", "Seleccione una carpeta de destino válida")
            return
        
        # Iniciar procesamiento para CDV
        self.start_processing("CDV")
        
        # Iniciar procesamiento para ADV
        self.start_processing("ADV")
    
    def process_velcom(self):
        """Procesar archivo Velcom"""
        # Verificar que se hayan seleccionado las rutas
        velcom_file = self.velcom_file_var.get()
        output_path = self.velcom_output_var.get()
        
        if not velcom_file or not os.path.exists(velcom_file):
            messagebox.showerror("Error", "Seleccione un archivo Velcom válido")
            return
        
        if not output_path or not os.path.exists(output_path):
            messagebox.showerror("Error", "Seleccione una carpeta de destino válida")
            return
        
        # Reiniciar barra de progreso
        self.velcom_progress_var.set(0)
        self.velcom_status_var.set("Iniciando procesamiento de Velcom...")
        self.view_velcom_button.config(state=tk.DISABLED)
        
        # Importar el procesador Velcom
        try:
            from processors.velcom_processor import VelcomProcessor
            processor = VelcomProcessor()
            processor.set_paths(velcom_file, output_path)
            processor.set_progress_callback(self.update_velcom_progress)
            
            # Iniciar procesamiento en un hilo separado
            processing_thread = threading.Thread(
                target=self._run_velcom_processing,
                args=(processor,)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
            self.log("Procesamiento de Velcom iniciado en segundo plano")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar procesamiento: {str(e)}")
            self.log(f"Error al iniciar procesamiento de Velcom: {str(e)}")

    def _run_velcom_processing(self, processor):
        """Ejecutar procesamiento Velcom en segundo plano"""
        try:
            success = processor.process_file()
            
            # Actualizar UI en el hilo principal
            self.frame.after(100, lambda: self._finish_velcom_processing(success))
            
        except Exception as e:
            # Manejar errores y actualizar UI en el hilo principal
            self.frame.after(
                100, 
                lambda: self.update_velcom_progress(0, f"Error en procesamiento: {str(e)}")
            )

    def _finish_velcom_processing(self, success):
        """Finalizar procesamiento Velcom"""
        if success:
            self.velcom_progress_var.set(100)
            self.velcom_status_var.set("Procesamiento de Velcom completado")
            self.view_velcom_button.config(state=tk.NORMAL)
            self.log("Procesamiento de Velcom completado con éxito")
        else:
            self.view_velcom_button.config(state=tk.DISABLED)
            self.log("Procesamiento de Velcom finalizado con errores")

    def update_velcom_progress(self, progress, message):
        """Actualizar progreso de procesamiento Velcom"""
        self.velcom_progress_var.set(progress)
        if message:
            self.velcom_status_var.set(message)
            self.log(f"[Velcom] {message}")

    def view_velcom_dashboard(self):
        """Mostrar dashboard de datos Velcom"""
        output_path = self.velcom_output_var.get()
        
        if not output_path or not os.path.exists(output_path):
            messagebox.showerror("Error", "No se puede acceder a la carpeta de resultados")
            return
        
        try:
            from dashboard.velcom_dashboard import launch_velcom_dashboard
            
            # Verificar que existan los archivos procesados
            required_files = [
                'velcom_data.csv',
                'velcom_trains.csv',
                'velcom_stations.csv',
                'velcom_info.csv'
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(output_path, file)):
                    messagebox.showerror(
                        "Error", 
                        f"Archivo de datos {file} no encontrado. Verifique que el procesamiento se haya completado correctamente."
                    )
                    return
            
            # Lanzar dashboard
            launch_velcom_dashboard(output_path)
            self.log("Dashboard de Velcom lanzado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al lanzar dashboard: {str(e)}")
            self.log(f"Error al lanzar dashboard de Velcom: {str(e)}")
    
    def update_progress(self, analysis_type, progress, message):
        """Actualizar barra de progreso y mensaje de estado"""
        if analysis_type == "CDV":
            if progress is not None:
                self.progress_var_cdv.set(progress)
            
            if message:
                self.status_var_cdv.set(message)
                self.log(f"[CDV] {message}")
                
            # Actualizar estado de procesamiento
            if progress == 100:
                self.cdv_processing_complete = True
                self.view_cdv_button.config(state=tk.NORMAL)
                self.log("[CDV] Procesamiento completo. Se puede visualizar el dashboard.")
        
        elif analysis_type == "ADV":
            if progress is not None:
                self.progress_var_adv.set(progress)
            
            if message:
                self.status_var_adv.set(message)
                self.log(f"[ADV] {message}")
                
            # Actualizar estado de procesamiento
            if progress == 100:
                self.adv_processing_complete = True
                self.view_adv_button.config(state=tk.NORMAL)
                self.log("[ADV] Procesamiento completo. Se puede visualizar el dashboard.")
    
    def view_results(self, analysis_type):
        """Visualizar resultados en dashboard web"""
        try:
            # Verificar si el procesamiento está completo
            if analysis_type == "CDV" and not self.cdv_processing_complete:
                messagebox.showwarning("Aviso", "El procesamiento de CDV no ha finalizado. No hay resultados para visualizar.")
                return
            
            if analysis_type == "ADV" and not self.adv_processing_complete:
                messagebox.showwarning("Aviso", "El procesamiento de ADV no ha finalizado. No hay resultados para visualizar.")
                return
            
            # Obtener la carpeta de destino
            dest_path = self.dest_path_var.get()
            if not dest_path or not os.path.exists(dest_path):
                messagebox.showerror("Error", "No se puede acceder a la carpeta de resultados.")
                return
            
            # Obtener la línea
            line = self.title.replace("Línea ", "L")
            
            # Inicializar el integrador de dashboard si aún no existe
            if self.dashboard_integration is None:
                self.dashboard_integration = DashboardIntegration(dest_path, self.frame.winfo_toplevel())
            
            # Lanzar el dashboard
            self.dashboard_integration.launch_dashboard(line, analysis_type)
            
            self.log(f"[{analysis_type}] Lanzando visualización de resultados...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar resultados: {str(e)}")
            self.log(f"Error al visualizar resultados de {analysis_type}: {str(e)}")