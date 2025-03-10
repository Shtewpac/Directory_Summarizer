from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLineEdit, QTextEdit, QLabel,
                           QFileDialog, QComboBox, QProgressBar, QDialog,
                           QDialogButtonBox, QInputDialog, QMessageBox,
                           QCheckBox, QTextBrowser)  # Add QTextBrowser
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCharFormat, QFont  # Add these imports
from pathlib import Path
from codebase_analyzer import (SimpleCodeAnalyzer, DirectoryAnalysis, 
                             save_analysis_to_file)  # Add save_analysis_to_file
from prompt_templates import PromptTemplates
import asyncio
import sys
import yaml
from datetime import datetime  # Add this import

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

    def __init__(self, directory, file_types, template_name, model="gpt-4o-mini"):
        super().__init__()
        self.directory = directory
        self.file_types = file_types
        self.template_name = template_name
        self.model = model
        self.perform_final_analysis = True  # Default value
        self.final_analysis_template = 'final_analysis'  # Default template
        self.final_analysis_model = None  # Add this line

    def run(self):
        try:
            analyzer = SimpleCodeAnalyzer(
                file_types=self.file_types,
                template_name=self.template_name,
                model=self.model,
                perform_final_analysis=self.perform_final_analysis,
                final_analysis_model=self.final_analysis_model,  # Pass the model
                progress_callback=self.progress_update.emit  # Add this line
            )
            # Set the final analysis template if enabled
            if self.perform_final_analysis:
                analyzer.final_analysis_prompt = analyzer.templates.get_template(
                    self.final_analysis_template
                )
            analysis = asyncio.run(analyzer.analyze_directory_async(self.directory))
            self.analysis_complete.emit(analysis)
        except Exception as e:
            self.error_occurred.emit(str(e))

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
        
        self.append(('\n'.join(lines)))

class CodeAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.ensure_output_directory()  # Add this line
        self.setWindowTitle("Code Analyzer")
        self.setMinimumSize(800, 600)
        self.analyzer_thread = None
        self.templates = PromptTemplates()
        self.setup_ui()

    def ensure_output_directory(self):
        """Create Sample_Outputs directory if it doesn't exist"""
        output_dir = Path(__file__).parent / "Sample_Outputs"
        output_dir.mkdir(exist_ok=True)
        self.config['paths']['output_dir'] = str(output_dir)
        self.config['paths']['output_file'] = str(output_dir / "code_analysis_report.md")

    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        dir_button = QPushButton("Select Directory")
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(QLabel("Directory:"))
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(dir_button)
        layout.addLayout(dir_layout)

        # Output configuration
        output_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        output_dir_button = QPushButton("Select Output Directory")
        output_dir_button.clicked.connect(self.select_output_directory)
        
        output_layout.addWidget(QLabel("Analysis Output Directory:"))
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(output_dir_button)
        layout.addLayout(output_layout)
        
        output_file_layout = QHBoxLayout()
        self.output_file_input = QLineEdit("code_analysis_report.md")
        self.final_output_file_input = QLineEdit("final_analysis.md")  # Keep this for filename only
        output_file_layout.addWidget(QLabel("Analysis Output File:"))
        output_file_layout.addWidget(self.output_file_input)
        output_file_layout.addWidget(QLabel("Final Analysis File:"))
        output_file_layout.addWidget(self.final_output_file_input)
        layout.addLayout(output_file_layout)

        # File types input
        file_layout = QHBoxLayout()
        self.file_types_input = QLineEdit("python,yaml,json")
        file_layout.addWidget(QLabel("File Types:"))
        file_layout.addWidget(self.file_types_input)
        layout.addLayout(file_layout)

        # Template selection with descriptions
        template_layout = QHBoxLayout()
        self.template_combo = TemplateComboBox()
        self.update_template_list()
        template_layout.addWidget(QLabel("Template:"))
        template_layout.addWidget(self.template_combo)
        
        # Add custom template button
        add_template_btn = QPushButton("Add Custom")
        add_template_btn.clicked.connect(self.add_custom_template)
        template_layout.addWidget(add_template_btn)
        
        # Add view/edit/delete buttons
        view_template_btn = QPushButton("View")
        edit_template_btn = QPushButton("Edit")
        delete_template_btn = QPushButton("Delete")  # New delete button
        view_template_btn.clicked.connect(self.view_template)
        edit_template_btn.clicked.connect(self.edit_template)
        delete_template_btn.clicked.connect(self.delete_template)  # Connect delete button
        template_layout.addWidget(view_template_btn)
        template_layout.addWidget(edit_template_btn)
        template_layout.addWidget(delete_template_btn)  # Add delete button
        layout.addLayout(template_layout)

        # Add model selection for file analysis
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o"])
        model_layout.addWidget(QLabel("File Analysis Model:"))
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Add model selection before final analysis controls
        final_model_layout = QHBoxLayout()
        self.final_model_combo = QComboBox()
        
        # Add available models from config
        for model_name in self.config['model']['configs'].keys():
            self.final_model_combo.addItem(model_name)
        
        final_model_layout.addWidget(QLabel("Final Analysis Model:"))
        final_model_layout.addWidget(self.final_model_combo)
        layout.addLayout(final_model_layout)

        # Add final analysis controls after template selection
        final_analysis_layout = QHBoxLayout()
        self.final_analysis_checkbox = QCheckBox("Enable Final Analysis")
        self.final_analysis_checkbox.setChecked(True)  # Enable by default
        self.final_analysis_checkbox.stateChanged.connect(self.toggle_final_analysis_template)
        final_analysis_layout.addWidget(self.final_analysis_checkbox)
        
        # Final analysis template selection
        self.final_template_combo = TemplateComboBox()
        self.final_template_combo.addItem("Default Final Analysis")
        self.update_final_template_list()
        
        # Add view/edit buttons for final templates
        final_view_btn = QPushButton("View Final")
        final_edit_btn = QPushButton("Edit Final")
        final_view_btn.clicked.connect(self.view_final_template)
        final_edit_btn.clicked.connect(self.edit_final_template)
        
        final_custom_btn = QPushButton("Custom Final")
        final_custom_btn.clicked.connect(self.add_custom_final_template)
        
        final_analysis_layout.addWidget(self.final_template_combo)
        final_analysis_layout.addWidget(final_view_btn)
        final_analysis_layout.addWidget(final_edit_btn)
        final_analysis_layout.addWidget(final_custom_btn)
        layout.addLayout(final_analysis_layout)
        
        # Initialize final template visibility
        self.toggle_final_analysis_template(True)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Add progress log text area
        self.progress_log = MarkdownTextEdit()
        self.progress_log.setMaximumHeight(100)  # Limit height
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_log)

        # Split the results area into two text boxes
        results_splitter = QVBoxLayout()
        
        # Main results view
        self.results_text = MarkdownTextEdit()  # Change to MarkdownTextEdit
        results_splitter.addWidget(QLabel("File Analysis Results:"))
        results_splitter.addWidget(self.results_text)

        # Final analysis view
        self.final_analysis_text = MarkdownTextEdit()  # Change to MarkdownTextEdit
        final_label = QLabel("Final Codebase Analysis:")
        results_splitter.addWidget(final_label)
        results_splitter.addWidget(self.final_analysis_text)

        layout.addLayout(results_splitter)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # Use config values for defaults
        output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir_input.setText(str(output_dir))
        self.output_file_input.setText("code_analysis_report.md")
        self.dir_input.setText(self.config['paths']['project_dir'])
        
        # Set final analysis checkbox based on config
        self.final_analysis_checkbox.setChecked(self.config['analysis']['perform_final_analysis'])

    def toggle_final_analysis_template(self, enabled):
        """Show/hide final analysis template controls"""
        self.final_template_combo.setVisible(enabled)
        self.final_template_combo.setEnabled(enabled)

    def update_template_list(self):
        """Update the template dropdown with current templates and descriptions"""
        self.template_combo.clear()
        for name, (template, description) in self.templates.get_templates_with_descriptions().items():
            self.template_combo.addItem(f"{name} - {description}")
            # Store the actual template name as item data
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
        template_name = self.get_selected_template_name()  # Get actual template name
        if not template_name:
            return
            
        template_text = self.templates.get_template(template_name)
        
        dialog = TemplateEditDialog(template_name, template_text, self)
        dialog.editor.setReadOnly(True)  # Make it read-only for viewing
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
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)

    def select_output_directory(self):
        """Select output directory for analysis report"""
        directory = QFileDialog.getExistingDirectory(self, "Select Analysis Output Directory")
        if directory:
            self.output_dir_input.setText(directory)

    def start_analysis(self):
        if not self.dir_input.text():
            self.results_text.setText("Please select a directory first.")
            return
            
        if not self.output_dir_input.text():
            self.results_text.setText("Please select an output directory first.")
            return

        selected_template = self.get_selected_template_name()
        if not self.templates.manager.template_exists(selected_template):
            self.results_text.setText(f"Template not found: {selected_template}")
            return

        # Create output file path
        output_file = Path(self.output_dir_input.text()) / self.output_file_input.text()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.show()
        self.progress_bar.setFormat("Analyzing...")
        self.progress_bar.setRange(0, 0)

        # Clear previous progress log
        self.progress_log.clear()
        self.progress_log.append_markdown("# Progress Log\n")

        self.analyzer_thread = AnalyzerThread(
            self.dir_input.text(),
            self.file_types_input.text(),
            selected_template,  # Use the actual template name
            self.model_combo.currentText()  # Pass selected model
        )
        
        # Pass output file path to analyzer thread
        self.analyzer_thread.output_file = str(output_file)
        
        # Configure final analysis settings
        self.analyzer_thread.perform_final_analysis = self.final_analysis_checkbox.isChecked()
        if self.analyzer_thread.perform_final_analysis:
            final_template = self.get_selected_final_template_name()
            self.analyzer_thread.final_analysis_template = final_template
            self.analyzer_thread.final_analysis_model = self.final_model_combo.currentText()
            
        # Pass the current templates to the analyzer thread
        self.analyzer_thread.analyzer_templates = self.templates
        self.analyzer_thread.analysis_complete.connect(self.handle_analysis_complete)
        self.analyzer_thread.error_occurred.connect(self.handle_error)
        self.analyzer_thread.progress_update.connect(self.update_progress)  # Connect progress updates
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
            analysis.results.remove(final_analysis)  # Remove from main results
            # Display final analysis in the dedicated text box
            self.final_analysis_text.append_markdown(final_analysis.analysis)
        
        # Save file analyses to main output
        output_file = Path(self.output_dir_input.text()) / self.output_file_input.text()
        save_analysis_to_file(analysis, str(output_file))
        self.results_text.append_markdown(f"\nResults saved to: {output_file}\n\n")
        
        # Show file analyses in main window with markdown
        for result in analysis.results:
            self.results_text.append_markdown(f"\n## {result.path}\n")
            self.results_text.append_markdown(result.analysis)
            self.results_text.append_markdown("\n---\n")

        # Save final analysis to same directory
        if final_analysis:
            final_output = Path(self.output_dir_input.text()) / self.final_output_file_input.text()
            with open(final_output, 'w', encoding='utf-8') as f:
                f.write("# Final Codebase Analysis\n\n")
                f.write(final_analysis.analysis)
            self.results_text.append_markdown(f"\nFinal analysis saved to: {final_output}\n")

        self.reset_ui()

    def handle_error(self, error_msg: str):
        self.results_text.setText(f"Error: {error_msg}")
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

    def reset_ui(self):
        self.progress_bar.hide()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # Add completion message to log
        self.update_progress("Analysis complete!")

def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = CodeAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()