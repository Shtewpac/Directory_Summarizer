from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QTextEdit, QLabel,
                             QFileDialog, QComboBox, QProgressBar, QDialog,
                             QDialogButtonBox, QInputDialog, QMessageBox,
                             QCheckBox, QTextBrowser, QListWidget, QListWidgetItem,
                             QGroupBox, QGridLayout, QToolButton, QApplication)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from pathlib import Path
from codebase_analyzer import (SimpleCodeAnalyzer, DirectoryAnalysis,
                               save_analysis_to_file)
from prompt_templates import PromptTemplates
import asyncio
import sys
import yaml
from datetime import datetime

class TemplateEditDialog(QDialog):
    def __init__(self, template_name: str, template_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Template - {template_name}")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        # Template editor
        self.editor = QTextEdit()
        self.editor.setPlainText(template_text)
        layout.addWidget(self.editor)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_template_text(self) -> str:
        return self.editor.toPlainText()

class TemplateComboBox(QComboBox):
    def showPopup(self):
        # Adjust width to fit descriptions
        width = self.minimumSizeHint().width()
        for i in range(self.count()):
            width = max(width, self.view().sizeHintForColumn(0))
        self.view().setMinimumWidth(width + 50)  # Add padding
        super().showPopup()

class AnalyzerThread(QThread):
    analysis_complete = pyqtSignal(DirectoryAnalysis)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    status_update = pyqtSignal(str)  # New signal for status updates
    file_types_detected = pyqtSignal(str)  # Add new signal

    def __init__(self, directory, file_types, template_name, model="gpt-4o-mini"):
        super().__init__()
        self.directory = directory
        self.file_types = file_types
        self.template_name = template_name
        self.model = model
        self.perform_final_analysis = True  # Default value
        self.final_analysis_template = 'final_analysis'  # Default template
        self.final_analysis_model = None
        self.progress_callback = self.update_progress  # Add progress callback

    def update_progress(self, message: str):
        self.progress_update.emit(message)

    def run(self):
        try:
            analyzer = SimpleCodeAnalyzer(
                file_types=self.file_types,
                template_name=self.template_name,
                model=self.model,
                perform_final_analysis=self.perform_final_analysis,
                final_analysis_model=self.final_analysis_model,
                progress_callback=self.progress_update.emit,
                file_types_callback=self.file_types_detected.emit  # Add callback
            )
            # Set the final analysis template if enabled
            if self.perform_final_analysis:
                analyzer.final_analysis_prompt = analyzer.templates.get_template(
                    self.final_analysis_template
                )
            self.status_update.emit("Starting analysis...")
            analysis = asyncio.run(analyzer.analyze_directory_async(self.directory))
            self.analysis_complete.emit(analysis)
            self.status_update.emit("Analysis complete")
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.status_update.emit("Analysis failed")

class MarkdownTextEdit(QTextBrowser):
    """Custom text edit widget for displaying markdown content"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)

        # Set up styling
        self.document().setDefaultStyleSheet("""
            h1 { font-size: 18pt; color: #2c3e50; margin-top: 20px; margin-bottom: 10px; }
            h2 { font-size: 16pt; color: #34495e; margin-top: 15px; margin-bottom: 8px; }
            code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
            strong { color: #2c3e50; }
            em { color: #7f8c8d; }
            hr { border: none; border-top: 1px solid #bdc3c7; margin: 15px 0; }
        """)

        # Set font
        font = QFont("Consolas", 10)
        self.setFont(font)

    def append_markdown(self, text: str):
        """Append markdown text with proper formatting"""
        # Basic markdown processing
        lines = []
        in_code_block = False

        for line in text.split('\n'):
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    lines.append('<pre><code>')
                else:
                    lines.append('</code></pre>')
                continue

            if in_code_block:
                lines.append(line)
                continue

            # Handle headings
            if line.startswith('# '):
                line = f'<h1>{line[2:]}</h1>'
            elif line.startswith('## '):
                line = f'<h2>{line[3:]}</h2>'

            # Handle bold text
            while '**' in line:
                line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)

            # Handle italic text
            while '*' in line:
                line = line.replace('*', '<em>', 1).replace('*', '</em>', 1)

            # Handle horizontal rules
            if line.strip() == '---':
                line = '<hr>'

            lines.append(line)

        self.append('\n'.join(lines))

class CodeAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set window icon - before any other initialization
        icon_path = str(Path(__file__).parent / "assets" / "code_analyzer.png")
        self.setWindowIcon(QIcon(icon_path))
        # Also set the taskbar icon for Windows
        if sys.platform == 'win32':
            import ctypes
            myappid = 'codeanalyzer.v6'  # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        
        self.config = self.load_config()
        self.ensure_output_directory()
        self.setWindowTitle("Code Analyzer")
        self.setMinimumSize(1000, 800)
        self.analyzer_thread = None
        self.templates = PromptTemplates()
        self.setup_ui()

    def ensure_output_directory(self):
        """Create Sample_Outputs directory and Sample_Directories if they don't exist"""
        # Create and set output directory
        output_dir = Path(__file__).parent / "Sample_Outputs"
        output_dir.mkdir(exist_ok=True)
        self.config['paths']['output_dir'] = str(output_dir)
        self.config['paths']['output_file'] = str(output_dir / "code_analysis_report.md")

        # Create and set project directory
        project_dir = Path(__file__).parent / "Sample_Directories"
        project_dir.mkdir(exist_ok=True)
        self.config['paths']['project_dir'] = str(project_dir)

    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)

        # Grouped sections using QGroupBox
        input_group = QGroupBox("Input Configuration")
        analysis_group = QGroupBox("Analysis Configuration")
        results_group = QGroupBox("Results")

        # Input Configuration
        input_layout = QGridLayout()
        input_group.setLayout(input_layout)

        # Directory selection
        self.dir_input = QLineEdit()
        dir_button = QPushButton("Select Directory")
        dir_button.clicked.connect(self.select_directory)
        self.dir_copy_button = QToolButton()
        self.dir_copy_button.setIcon(QIcon.fromTheme("edit-copy"))
        self.dir_copy_button.clicked.connect(lambda: self.copy_to_clipboard(self.dir_input.text()))
        self.dir_input.setPlaceholderText("Select a directory to analyze")
        self.dir_input.setToolTip("Select a directory to analyze")
        input_layout.addWidget(QLabel("Directory:"), 0, 0)
        dir_input_layout = QHBoxLayout()
        dir_input_layout.addWidget(self.dir_input)
        dir_input_layout.addWidget(self.dir_copy_button)
        dir_input_layout.addWidget(dir_button)
        input_layout.addLayout(dir_input_layout, 0, 1)

        # Output configuration
        self.output_dir_input = QLineEdit()
        output_dir_button = QPushButton("Select Output Directory")
        output_dir_button.clicked.connect(self.select_output_directory)
        self.output_dir_copy_button = QToolButton()
        self.output_dir_copy_button.setIcon(QIcon.fromTheme("edit-copy"))
        self.output_dir_copy_button.clicked.connect(lambda: self.copy_to_clipboard(self.output_dir_input.text()))
        self.output_dir_input.setPlaceholderText("Select a directory for output")
        self.output_dir_input.setToolTip("Select a directory for output files")
        input_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        output_dir_input_layout = QHBoxLayout()
        output_dir_input_layout.addWidget(self.output_dir_input)
        output_dir_input_layout.addWidget(self.output_dir_copy_button)
        output_dir_input_layout.addWidget(output_dir_button)
        input_layout.addLayout(output_dir_input_layout, 1, 1)

        # Output file name
        self.output_file_input = QLineEdit("code_analysis_report.md")
        self.final_output_file_input = QLineEdit("final_analysis.md")
        input_layout.addWidget(QLabel("Output File:"), 2, 0)
        input_layout.addWidget(self.output_file_input, 2, 1)
        input_layout.addWidget(QLabel("Final Analysis File:"), 3, 0)
        input_layout.addWidget(self.final_output_file_input, 3, 1)

        # Replace file types list with text input
        self.file_types_input = QLineEdit()
        self.file_types_input.setPlaceholderText("Describe file types, or leave empty for automatic detection")
        self.file_types_input.setToolTip("Enter file types to analyze, or leave empty to automatically detect relevant files")
        input_layout.addWidget(QLabel("File Types:"), 4, 0)
        input_layout.addWidget(self.file_types_input, 4, 1)

        # Add image analysis option with supported formats
        self.analyze_images_checkbox = QCheckBox("Analyze Images")
        self.analyze_images_checkbox.setToolTip(
            "Include image analysis for formats:\n"
            "- PNG\n"
            "- JPEG/JPG\n"
            "- GIF\n"
            "- BMP\n"
            "- SVG\n"
            "- WEBP"
        )
        input_layout.addWidget(QLabel("Options:"), 5, 0)
        input_layout.addWidget(self.analyze_images_checkbox, 5, 1)

        main_layout.addWidget(input_group)

        # Analysis Configuration
        analysis_layout = QGridLayout()
        analysis_group.setLayout(analysis_layout)

        # Template selection with descriptions
        self.template_combo = TemplateComboBox()
        self.update_template_list()
        self.template_combo.setToolTip("Select an analysis template")
        self.template_combo.currentIndexChanged.connect(self.show_template_tooltip)

        analysis_layout.addWidget(QLabel("Template:"), 0, 0)
        template_layout = QHBoxLayout()
        template_layout.addWidget(self.template_combo)

        # Add custom template button
        add_template_btn = QPushButton("Add Custom")
        add_template_btn.clicked.connect(self.add_custom_template)
        template_layout.addWidget(add_template_btn)

        # Add view/edit/delete buttons
        view_template_btn = QPushButton("View")
        edit_template_btn = QPushButton("Edit")
        delete_template_btn = QPushButton("Delete")
        view_template_btn.clicked.connect(self.view_template)
        edit_template_btn.clicked.connect(self.edit_template)
        delete_template_btn.clicked.connect(self.delete_template)
        template_layout.addWidget(view_template_btn)
        template_layout.addWidget(edit_template_btn)
        template_layout.addWidget(delete_template_btn)
        analysis_layout.addLayout(template_layout, 0, 1)

        # Model selection with tooltips
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o"])
        self.model_combo.setToolTip("Select the model for file analysis")
        self.model_combo.currentIndexChanged.connect(self.show_model_tooltip)
        analysis_layout.addWidget(QLabel("File Analysis Model:"), 1, 0)
        analysis_layout.addWidget(self.model_combo, 1, 1)

        # Final analysis settings
        self.final_analysis_checkbox = QCheckBox("Enable Final Analysis")
        self.final_analysis_checkbox.setChecked(True)
        self.final_analysis_checkbox.stateChanged.connect(self.toggle_final_analysis_template)
        analysis_layout.addWidget(self.final_analysis_checkbox, 2, 0)

        self.final_template_combo = TemplateComboBox()
        self.final_template_combo.addItem("Default Final Analysis")
        self.update_final_template_list()
        self.final_template_combo.setToolTip("Select a template for final analysis")
        self.final_template_combo.currentIndexChanged.connect(self.show_final_template_tooltip)

        # Final analysis controls
        final_analysis_layout = QHBoxLayout()
        final_analysis_layout.addWidget(self.final_template_combo)
        final_view_btn = QPushButton("View Final")
        final_edit_btn = QPushButton("Edit Final")
        final_view_btn.clicked.connect(self.view_final_template)
        final_edit_btn.clicked.connect(self.edit_final_template)
        final_custom_btn = QPushButton("Custom Final")
        final_custom_btn.clicked.connect(self.add_custom_final_template)
        final_analysis_layout.addWidget(final_view_btn)
        final_analysis_layout.addWidget(final_edit_btn)
        final_analysis_layout.addWidget(final_custom_btn)
        analysis_layout.addLayout(final_analysis_layout, 2, 1)

        # Final analysis model selection
        self.final_model_combo = QComboBox()
        for model_name in self.config['model']['configs'].keys():
            self.final_model_combo.addItem(model_name)
        self.final_model_combo.setToolTip("Select the model for final analysis")
        self.final_model_combo.currentIndexChanged.connect(self.show_final_model_tooltip)
        analysis_layout.addWidget(QLabel("Final Analysis Model:"), 3, 0)
        analysis_layout.addWidget(self.final_model_combo, 3, 1)

        main_layout.addWidget(analysis_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        main_layout.addLayout(control_layout)

        # Results Section
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        # Progress bar and log
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        results_layout.addWidget(self.progress_bar)

        self.progress_log = MarkdownTextEdit()
        self.progress_log.setMaximumHeight(100)
        results_layout.addWidget(QLabel("Progress:"))
        results_layout.addWidget(self.progress_log)

        # Results text areas
        self.results_text = MarkdownTextEdit()
        results_layout.addWidget(QLabel("File Analysis Results:"))
        results_layout.addWidget(self.results_text)

        self.final_analysis_text = MarkdownTextEdit()
        results_layout.addWidget(QLabel("Final Codebase Analysis:"))
        results_layout.addWidget(self.final_analysis_text)

        main_layout.addWidget(results_group)

        # Use config values for defaults
        output_dir = Path(self.config['paths']['output_dir'])
        project_dir = Path(self.config['paths']['project_dir'])
        self.output_dir_input.setText(str(output_dir))
        self.output_file_input.setText("code_analysis_report.md")
        self.dir_input.setText(str(project_dir))

        # Set final analysis checkbox based on config
        self.final_analysis_checkbox.setChecked(self.config['analysis']['perform_final_analysis'])

        # Initialize final template visibility
        self.toggle_final_analysis_template(True)

    def copy_to_clipboard(self, text: str):
        """Copy text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def show_template_tooltip(self):
        """Show tooltip for selected template"""
        template_name = self.get_selected_template_name()
        template = self.templates.manager.templates.get(template_name)
        if template:
            self.template_combo.setToolTip(template.description)

    def show_final_template_tooltip(self):
        """Show tooltip for selected final template"""
        template_name = self.get_selected_final_template_name()
        template = self.templates.manager.final_templates.get(template_name)
        if template:
            self.final_template_combo.setToolTip(template.description)

    def show_model_tooltip(self):
        """Show tooltip for selected model"""
        model_name = self.model_combo.currentText()
        model_info = self.config['model']['configs'].get(model_name, {})
        tooltip = f"Context Window: {model_info.get('context_window', 'N/A')} tokens"
        self.model_combo.setToolTip(tooltip)

    def show_final_model_tooltip(self):
        """Show tooltip for selected final model"""
        model_name = self.final_model_combo.currentText()
        model_info = self.config['model']['configs'].get(model_name, {})
        tooltip = f"Context Window: {model_info.get('context_window', 'N/A')} tokens"
        self.final_model_combo.setToolTip(tooltip)

    def toggle_final_analysis_template(self, enabled):
        """Show/hide final analysis template controls"""
        self.final_template_combo.setVisible(enabled)
        self.final_template_combo.setEnabled(enabled)

    def update_template_list(self):
        """Update the template dropdown with current templates and descriptions"""
        self.template_combo.clear()
        for name, (template, description) in self.templates.get_templates_with_descriptions().items():
            self.template_combo.addItem(f"{name} - {description}")
            self.template_combo.setItemData(self.template_combo.count() - 1, name)

    def update_final_template_list(self):
        """Update the final analysis template dropdown"""
        self.final_template_combo.clear()
        self.final_template_combo.addItem("Default Final Analysis", "final_analysis")
        for name, (_, description) in self.templates.get_final_templates_with_descriptions().items():
            self.final_template_combo.addItem(f"{name} - {description}")
            self.final_template_combo.setItemData(
                self.final_template_combo.count() - 1,
                name
            )

    def get_selected_template_name(self) -> str:
        """Get the actual template name without description"""
        return self.template_combo.currentData()

    def get_selected_final_template_name(self) -> str:
        """Get the actual final template name without description"""
        return self.final_template_combo.currentData() or "final_analysis"

    def add_custom_template(self):
        """Add a new custom template"""
        name, ok = QInputDialog.getText(self, "New Template", "Enter template name:")
        if ok and name:
            if name in self.templates.manager.templates:
                QMessageBox.warning(self, "Error", "Template name already exists!")
                return

            dialog = TemplateEditDialog(name, "", self)
            dialog.setWindowTitle("Create Custom Template")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Save as a user template
                self.templates.manager.add_template(
                    name,
                    dialog.get_template_text(),
                    "Custom template"
                )
                self.update_template_list()
                index = self.template_combo.findData(name)
                if index >= 0:
                    self.template_combo.setCurrentIndex(index)

    def add_custom_final_template(self):
        """Add a custom template specifically for final analysis"""
        name, ok = QInputDialog.getText(self, "New Final Analysis Template",
                                        "Enter template name:")
        if ok and name:
            if name in self.templates.manager.templates:
                QMessageBox.warning(self, "Error", "Template name already exists!")
                return

            default_content = self.templates.manager.get_template('final_analysis')
            dialog = TemplateEditDialog(name, default_content, self)
            dialog.setWindowTitle("Create Custom Final Analysis Template")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Save as a user template
                self.templates.manager.add_template(
                    name,
                    dialog.get_template_text(),
                    "Custom final analysis template"
                )
                self.update_template_list()
                self.update_final_template_list()

    def view_template(self):
        """Show the current template content"""
        template_name = self.get_selected_template_name()
        if not template_name:
            return

        template_text = self.templates.get_template(template_name)

        dialog = TemplateEditDialog(template_name, template_text, self)
        dialog.editor.setReadOnly(True)
        dialog.setWindowTitle(f"View Template - {template_name}")
        dialog.exec()

    def view_final_template(self):
        """Show the current final analysis template content"""
        template_name = self.get_selected_final_template_name()
        template_text = self.templates.get_template(template_name, True)

        dialog = TemplateEditDialog(template_name, template_text, self)
        dialog.editor.setReadOnly(True)
        dialog.setWindowTitle(f"View Final Template - {template_name}")
        dialog.exec()

    def edit_template(self):
        """Edit the current template"""
        template_name = self.get_selected_template_name()
        if not template_name:
            return

        template = self.templates.manager.templates.get(template_name)
        if not template:
            return

        dialog = TemplateEditDialog(template_name, template.content, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_text = dialog.get_template_text()
            # Update and save the template
            template.content = new_text
            self.templates.manager.save_template(template_name, template)
            self.update_template_list()

    def edit_final_template(self):
        """Edit the current final analysis template"""
        template_name = self.get_selected_final_template_name()
        if not template_name:
            return

        template = self.templates.manager.templates.get(template_name)
        if not template:
            return

        dialog = TemplateEditDialog(template_name, template.content, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_text = dialog.get_template_text()
            # Update and save the template
            template.content = new_text
            self.templates.manager.save_template(template_name, template)
            self.update_final_template_list()

    def delete_template(self):
        """Delete the currently selected template"""
        template_name = self.get_selected_template_name()
        if not template_name:
            return

        template = self.templates.manager.templates.get(template_name)
        if not template:
            return

        # Don't allow deletion of system templates
        if template.author.lower() == 'system':
            QMessageBox.warning(
                self,
                "Cannot Delete Template",
                "System templates cannot be deleted."
            )
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the template '{template_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Delete the template file and remove from manager
            try:
                self.templates.manager.delete_template(template_name)
                self.update_template_list()
                self.update_final_template_list()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete template: {str(e)}"
                )

    def select_directory(self):
        # Use Sample_Directories as the starting directory
        start_dir = str(Path(__file__).parent / "Sample_Directories")
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory",
            start_dir
        )
        if directory:
            self.dir_input.setText(self.truncate_path(directory))
            self.dir_input.setToolTip(directory)

    def select_output_directory(self):
        """Select output directory for analysis report"""
        start_dir = str(Path(__file__).parent / "Sample_Outputs")
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Analysis Output Directory",
            start_dir
        )
        if directory:
            self.output_dir_input.setText(self.truncate_path(directory))
            self.output_dir_input.setToolTip(directory)

    def truncate_path(self, path: str, max_length: int = 50) -> str:
        """Truncate long paths with ellipsis in the middle"""
        if len(path) <= max_length:
            return path
        else:
            return f"{path[:25]}...{path[-25:]}"

    def get_selected_file_types(self) -> str:
        """Get file types description from text input"""
        return self.file_types_input.text()

    def start_analysis(self):
        # Check for valid output directory
        output_dir = self.output_dir_input.toolTip()
        if not output_dir or output_dir == "Select a directory for output files":
            # Use default Sample_Outputs directory if none selected
            output_dir = str(Path(__file__).parent / "Sample_Outputs")
            self.output_dir_input.setToolTip(output_dir)
            self.output_dir_input.setText(self.truncate_path(output_dir))

        if not self.dir_input.text():
            self.results_text.setText("Please select a directory first.")
            return

        selected_template = self.get_selected_template_name()
        if not self.templates.manager.template_exists(selected_template):
            self.results_text.setText(f"Template not found: {selected_template}")
            return

        # Create output file path using the actual directory path
        output_file = Path(output_dir) / self.output_file_input.text()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.show()
        self.progress_bar.setFormat("Analyzing...")
        self.progress_bar.setRange(0, 0)

        # Clear previous progress log
        self.progress_log.clear()
        self.progress_log.append_markdown("# Progress Log\n")

        self.analyzer_thread = AnalyzerThread(
            self.dir_input.toolTip(),
            self.get_selected_file_types() or None,  # Pass None if empty
            selected_template,
            self.model_combo.currentText()
        )

        # Add image analysis setting
        self.analyzer_thread.analyze_images = self.analyze_images_checkbox.isChecked()

        # Configure final analysis settings
        self.analyzer_thread.perform_final_analysis = self.final_analysis_checkbox.isChecked()
        if self.analyzer_thread.perform_final_analysis:
            final_template = self.get_selected_final_template_name()
            self.analyzer_thread.final_analysis_template = final_template
            self.analyzer_thread.final_analysis_model = self.final_model_combo.currentText()

        self.analyzer_thread.analysis_complete.connect(self.handle_analysis_complete)
        self.analyzer_thread.error_occurred.connect(self.handle_error)
        self.analyzer_thread.progress_update.connect(self.update_progress)
        self.analyzer_thread.status_update.connect(self.update_status)
        self.analyzer_thread.file_types_detected.connect(self.update_file_types_input)
        self.analyzer_thread.start()

    def stop_analysis(self):
        if self.analyzer_thread and self.analyzer_thread.isRunning():
            self.analyzer_thread.terminate()
            self.analyzer_thread.wait()
            self.reset_ui()

    def handle_analysis_complete(self, analysis: DirectoryAnalysis):
        self.results_text.clear()
        self.final_analysis_text.clear()

        # Update to use append_markdown
        self.results_text.append_markdown(f"# Analysis Results\n")
        self.results_text.append_markdown(f"**Directory:** {analysis.directory}\n")
        self.results_text.append_markdown(f"**Files analyzed:** {analysis.file_count}\n")

        # Extract final analysis before saving file analyses
        final_analysis = next((r for r in analysis.results if r.path == "FINAL_ANALYSIS"), None)
        if final_analysis:
            analysis.results.remove(final_analysis)
            self.final_analysis_text.append_markdown(final_analysis.analysis)

        # Save file analyses to main output
        output_file = Path(self.output_dir_input.toolTip()) / self.output_file_input.text()
        save_analysis_to_file(analysis, str(output_file))
        self.results_text.append_markdown(f"\nResults saved to: {output_file}\n\n")

        # Show file analyses in main window with markdown
        for result in analysis.results:
            self.results_text.append_markdown(f"\n## {result.path}\n")
            self.results_text.append_markdown(result.analysis)
            self.results_text.append_markdown("\n---\n")

        # Save final analysis to same directory
        if final_analysis:
            final_output = Path(self.output_dir_input.toolTip()) / self.final_output_file_input.text()
            with open(final_output, 'w', encoding='utf-8') as f:
                f.write("# Final Codebase Analysis\n\n")
                f.write(final_analysis.analysis)
            self.results_text.append_markdown(f"\nFinal analysis saved to: {final_output}\n")

        self.reset_ui()

    def handle_error(self, error_msg: str):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        self.reset_ui()

    def update_progress(self, message: str):
        """Update progress bar and log with current status"""
        # Update progress bar text
        self.progress_bar.setFormat(f"Analyzing... {message}")

        # Add message to log with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_log.append_markdown(f"**{timestamp}:** {message}")

        # Scroll to bottom to show latest message
        scrollbar = self.progress_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_status(self, status: str):
        """Update the status bar with current status"""
        self.status_label.setText(status)

    def update_file_types_input(self, file_types: str):
        """Update the file types input field with detected types"""
        self.file_types_input.setText(file_types)

    def reset_ui(self):
        self.progress_bar.hide()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Ready")
        self.update_progress("Analysis complete!")

def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = CodeAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()