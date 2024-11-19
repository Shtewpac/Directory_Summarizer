from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLineEdit, QTextEdit, QLabel,
                           QFileDialog, QComboBox, QProgressBar, QDialog,
                           QDialogButtonBox, QInputDialog, QMessageBox,
                           QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from codebase_analyzer import (SimpleCodeAnalyzer, DirectoryAnalysis, 
                             save_analysis_to_file)  # Add save_analysis_to_file
from prompt_templates import PromptTemplates
import asyncio
import sys
import yaml

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

    def __init__(self, directory, file_types, template_name):
        super().__init__()
        self.directory = directory
        self.file_types = file_types
        self.template_name = template_name
        self.perform_final_analysis = True  # Default value
        self.final_analysis_template = 'final_analysis'  # Default template

    def run(self):
        try:
            analyzer = SimpleCodeAnalyzer(
                file_types=self.file_types,
                template_name=self.template_name,
                perform_final_analysis=self.perform_final_analysis
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

class CodeAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.setWindowTitle("Code Analyzer")
        self.setMinimumSize(800, 600)
        self.analyzer_thread = None
        self.templates = PromptTemplates()  # Add this line to store templates
        self.setup_ui()

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

        # Add output file configuration after directory selection
        output_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.output_file_input = QLineEdit("code_analysis_report.md")
        output_dir_button = QPushButton("Select Output Directory")
        output_dir_button.clicked.connect(self.select_output_directory)
        
        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(output_dir_button)
        layout.addLayout(output_layout)
        
        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(QLabel("Output Filename:"))
        output_file_layout.addWidget(self.output_file_input)
        layout.addLayout(output_file_layout)

        # Add separate final analysis output configuration
        final_output_layout = QHBoxLayout()
        self.final_output_dir_input = QLineEdit()
        self.final_output_file_input = QLineEdit("final_analysis.md")
        final_output_dir_button = QPushButton("Select Final Analysis Output Directory")
        final_output_dir_button.clicked.connect(self.select_final_output_directory)
        
        final_output_layout.addWidget(QLabel("Final Analysis Output:"))
        final_output_layout.addWidget(self.final_output_dir_input)
        final_output_layout.addWidget(final_output_dir_button)
        layout.addLayout(final_output_layout)
        
        final_output_file_layout = QHBoxLayout()
        final_output_file_layout.addWidget(QLabel("Final Analysis Filename:"))
        final_output_file_layout.addWidget(self.final_output_file_input)
        layout.addLayout(final_output_file_layout)

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
        
        # Add view/edit buttons
        view_template_btn = QPushButton("View")
        edit_template_btn = QPushButton("Edit")
        view_template_btn.clicked.connect(self.view_template)
        edit_template_btn.clicked.connect(self.edit_template)
        template_layout.addWidget(view_template_btn)
        template_layout.addWidget(edit_template_btn)
        layout.addLayout(template_layout)

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

        # Split the results area into two text boxes
        results_splitter = QVBoxLayout()
        
        # Main results view
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_splitter.addWidget(QLabel("File Analysis Results:"))
        results_splitter.addWidget(self.results_text)

        # Final analysis view
        self.final_analysis_text = QTextEdit()
        self.final_analysis_text.setReadOnly(True)
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
        self.output_dir_input.setText(str(Path(self.config['paths']['output_file']).parent))
        self.output_file_input.setText(Path(self.config['paths']['output_file']).name)
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
        for name, (_, description) in self.templates.get_templates_with_descriptions().items():
            if "final" in name.lower() or "final" in description.lower():
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
            # Check if template already exists
            if name in self.templates.templates:
                QMessageBox.warning(self, "Error", "Template name already exists!")
                return
                
            dialog = TemplateEditDialog(name, "", self)
            dialog.setWindowTitle("Create Custom Template")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.templates.add_template(
                    name, 
                    dialog.get_template_text(),
                    "Custom template"  # Default description for custom templates
                )
                self.update_template_list()
                # Select the new template
                index = self.template_combo.findData(name)
                if (index >= 0):
                    self.template_combo.setCurrentIndex(index)

    def add_custom_final_template(self):
        """Add a custom template specifically for final analysis"""
        name, ok = QInputDialog.getText(self, "New Final Analysis Template", 
                                      "Enter template name:")
        if ok and name:
            if name in self.templates.templates:
                QMessageBox.warning(self, "Error", "Template name already exists!")
                return

            dialog = TemplateEditDialog(name, self.templates.get_template('final_analysis'), self)
            dialog.setWindowTitle("Create Custom Final Analysis Template")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.templates.add_template(
                    name,
                    dialog.get_template_text(),
                    "Custom final analysis template"
                )
                # Update both template combos
                self.update_template_list()
                # Add to final analysis combo
                self.final_template_combo.addItem(f"{name} - Custom final analysis template")
                self.final_template_combo.setItemData(
                    self.final_template_combo.count() - 1,
                    name
                )

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
        template_text = self.templates.get_template(template_name)
        
        dialog = TemplateEditDialog(template_name, template_text, self)
        dialog.editor.setReadOnly(True)
        dialog.setWindowTitle(f"View Final Template - {template_name}")
        dialog.exec()

    def edit_template(self):
        """Edit the current template"""
        template_name = self.template_combo.currentText()
        template_text = self.templates.get_template(template_name)
        
        dialog = TemplateEditDialog(template_name, template_text, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_text = dialog.get_template_text()
            self.templates.templates[template_name] = new_text
            # Optionally save to file if template was loaded from one
            if hasattr(self.templates, 'template_path') and self.templates.template_path:
                self.templates.save_templates(self.templates.template_path)

    def edit_final_template(self):
        """Edit the current final analysis template"""
        template_name = self.get_selected_final_template_name()
        template_text = self.templates.get_template(template_name)
        
        dialog = TemplateEditDialog(template_name, template_text, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_text = dialog.get_template_text()
            self.templates.templates[template_name] = new_text
            if hasattr(self.templates, 'template_path') and self.templates.template_path:
                self.templates.save_templates(self.templates.template_path)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)

    def select_output_directory(self):
        """Select output directory for analysis report"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_input.setText(directory)

    def select_final_output_directory(self):
        """Select output directory for final analysis"""
        directory = QFileDialog.getExistingDirectory(self, "Select Final Analysis Output Directory")
        if directory:
            self.final_output_dir_input.setText(directory)

    def start_analysis(self):
        if not self.dir_input.text():
            self.results_text.setText("Please select a directory first.")
            return
            
        if not self.output_dir_input.text():
            self.results_text.setText("Please select an output directory first.")
            return
            
        if self.final_analysis_checkbox.isChecked() and not self.final_output_dir_input.text():
            self.results_text.setText("Please select a final analysis output directory.")
            return

        # Create output file path
        output_file = Path(self.output_dir_input.text()) / self.output_file_input.text()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.show()
        self.progress_bar.setFormat("Analyzing...")
        self.progress_bar.setRange(0, 0)

        self.analyzer_thread = AnalyzerThread(
            self.dir_input.text(),
            self.file_types_input.text(),
            self.get_selected_template_name()  # Use the actual template name
        )
        
        # Pass output file path to analyzer thread
        self.analyzer_thread.output_file = str(output_file)
        
        # Configure final analysis settings
        self.analyzer_thread.perform_final_analysis = self.final_analysis_checkbox.isChecked()
        if self.analyzer_thread.perform_final_analysis:
            final_template = self.get_selected_final_template_name()
            self.analyzer_thread.final_analysis_template = final_template
            
        # Pass the current templates to the analyzer thread
        self.analyzer_thread.analyzer_templates = self.templates
        self.analyzer_thread.analysis_complete.connect(self.handle_analysis_complete)
        self.analyzer_thread.error_occurred.connect(self.handle_error)
        self.analyzer_thread.start()

    def stop_analysis(self):
        if self.analyzer_thread and self.analyzer_thread.isRunning():
            self.analyzer_thread.terminate()
            self.analyzer_thread.wait()
            self.reset_ui()

    def handle_analysis_complete(self, analysis: DirectoryAnalysis):
        self.results_text.clear()
        self.final_analysis_text.clear()
        
        self.results_text.append(f"Analysis Results:\n")
        self.results_text.append(f"Directory: {analysis.directory}")
        self.results_text.append(f"Files analyzed: {analysis.file_count}\n")
        
        # Extract final analysis before saving file analyses
        final_analysis = next((r for r in analysis.results if r.path == "FINAL_ANALYSIS"), None)
        if final_analysis:
            analysis.results.remove(final_analysis)  # Remove from main results
            # Display final analysis in the dedicated text box
            self.final_analysis_text.setText(final_analysis.analysis)
        
        # Save file analyses to main output
        output_file = Path(self.output_dir_input.text()) / self.output_file_input.text()
        save_analysis_to_file(analysis, str(output_file))
        self.results_text.append(f"\nResults saved to: {output_file}\n\n")
        
        # Show file analyses in main window
        for result in analysis.results:
            self.results_text.append(f"\n--- {result.path} ---\n")
            self.results_text.append(result.analysis)
            self.results_text.append("\n" + "-"*50 + "\n")

        # Handle final analysis file saving if directory specified
        if final_analysis and self.final_output_dir_input.text():
            final_output = Path(self.final_output_dir_input.text()) / self.final_output_file_input.text()
            with open(final_output, 'w', encoding='utf-8') as f:
                f.write("# Final Codebase Analysis\n\n")
                f.write(final_analysis.analysis)
            self.results_text.append(f"\nFinal analysis saved to: {final_output}\n")

        self.reset_ui()

    def handle_error(self, error_msg: str):
        self.results_text.setText(f"Error: {error_msg}")
        self.reset_ui()

    def reset_ui(self):
        self.progress_bar.hide()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = CodeAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()