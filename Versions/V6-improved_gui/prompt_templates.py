from pathlib import Path
import yaml
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class Template:
    name: str
    version: str
    author: str
    description: str
    content: str
    extends: Optional[str] = None
    variables: Dict[str, str] = None

class TemplateManager:
    def __init__(self, template_dir: str = "templates"):
        # Always resolve template_dir relative to the current file's location
        self.template_dir = Path(__file__).parent / template_dir
        self.user_template_dir = self.template_dir / "user"
        self.templates: Dict[str, Template] = {}
        self.final_templates: Dict[str, Template] = {}
        self.ensure_directories()
        self.load_default_templates()  # First load built-in defaults
        self.load_all_templates()      # Then load from files

    def ensure_directories(self):
        """Create template directories if they don't exist"""
        self.template_dir.mkdir(exist_ok=True)
        self.user_template_dir.mkdir(exist_ok=True)
        # Create a .gitkeep file in the user directory to ensure it's tracked
        (self.user_template_dir / ".gitkeep").touch(exist_ok=True)

    def load_default_templates(self):
        """Load built-in default templates from files"""
        system_templates_dir = self.template_dir / "system"
        system_templates_dir.mkdir(exist_ok=True)

        # Ensure default template files exist
        self.ensure_default_template_files(system_templates_dir)

        # Load templates from system directory
        if system_templates_dir.exists():
            for yaml_file in system_templates_dir.glob("*.yaml"):
                self.load_template_file(yaml_file)

    def ensure_default_template_files(self, system_dir: Path):
        """Create default template files if they don't exist"""
        default_templates = {
            'analysis.yaml': {
                'analysis': {
                    'version': '1.0',
                    'author': 'System',
                    'description': 'General file analysis and structure',
                    'content': """
                        For the provided file:
                        1. File type and format overview
                        2. Key content summary
                        3. Structure and organization
                        4. Notable patterns or elements
                        5. Potential concerns or improvements
                    """
                }
            },
            'final_analysis.yaml': {
                'final_analysis': {
                    'version': '1.0',
                    'author': 'System',
                    'description': 'Overall directory summary and insights',
                    'content': """
                        Provide a comprehensive directory analysis:
                        1. Overview of files and their purposes
                        2. Common patterns across files
                        3. Key observations and findings
                        4. Organization assessment
                        5. Improvement suggestions
                    """
                }
            }
        }

        for filename, content in default_templates.items():
            template_file = system_dir / filename
            if not template_file.exists():
                with open(template_file, 'w') as f:
                    yaml.dump(content, f)

    def load_all_templates(self):
        """Load both system and user templates"""
        # Load system templates
        if self.template_dir.exists():
            for yaml_file in self.template_dir.glob("*.yaml"):
                self.load_template_file(yaml_file)

        # Load user templates
        if self.user_template_dir.exists():
            for yaml_file in self.user_template_dir.glob("*.yaml"):
                self.load_template_file(yaml_file)

    def load_template_file(self, file_path: Path):
        """Load templates from a YAML file"""
        try:
            with open(file_path, 'r') as f:
                templates = yaml.safe_load(f)
                for name, data in templates.items():
                    template = Template(
                        name=name,
                        version=data.get('version', '1.0'),
                        author=data.get('author', 'System'),
                        description=data.get('description', ''),
                        content=data['content'],
                        extends=data.get('extends'),
                        variables=data.get('variables', {})
                    )
                    # Store in appropriate dictionary based on name/type
                    if 'final' in name.lower() or 'final' in data.get('description', '').lower():
                        self.final_templates[name] = template
                    else:
                        self.templates[name] = template
        except Exception as e:
            logging.error(f"Error loading template file {file_path}: {e}")

    def save_template(self, name: str, template: Template, is_user_template: bool = True):
        """Save a template to the appropriate directory"""
        save_dir = self.user_template_dir if is_user_template else self.template_dir
        file_path = save_dir / f"{name}.yaml"

        template_data = {
            name: {
                'version': template.version,
                'author': template.author,
                'description': template.description,
                'content': template.content,
                'extends': template.extends,
                'variables': template.variables
            }
        }

        with open(file_path, 'w') as f:
            yaml.dump(template_data, f)
        logging.info(f"Saved template {name} to {file_path}")

    def get_template(self, name: str, is_final: bool = False) -> str:
        """Get processed template content with inheritance and variable substitution"""
        template_dict = self.final_templates if is_final else self.templates
        template = template_dict.get(name)
        if not template:
            raise KeyError(f"Template not found: {name}")

        content = template.content

        # Handle inheritance
        if template.extends:
            base_template = self.get_template(template.extends)
            content = self._apply_variables(base_template, template.variables)

        return content

    def _apply_variables(self, content: str, variables: Dict[str, str]) -> str:
        """Replace template variables with their values"""
        if not variables:
            return content

        for key, value in variables.items():
            content = content.replace(f"{{{{%s}}}}" % key, value)

        return content

    def list_templates(self, include_final: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get all templates with their metadata"""
        templates = self.templates.copy()
        if include_final:
            templates.update(self.final_templates)
        return {
            name: {
                'version': template.version,
                'author': template.author,
                'description': template.description,
                'extends': template.extends
            }
            for name, template in templates.items()
        }

    def list_final_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get only final analysis templates"""
        return {
            name: {
                'version': template.version,
                'author': template.author,
                'description': template.description,
                'extends': template.extends
            }
            for name, template in self.final_templates.items()
        }

    def add_template(self, name: str, content: str, description: str,
                     extends: Optional[str] = None, variables: Dict[str, str] = None):
        """Add a new template and save it"""
        template = Template(
            name=name,
            version='1.0',
            author='User',
            description=description,
            content=content,
            extends=extends,
            variables=variables or {}
        )
        self.templates[name] = template
        self.save_template(name, template)

    def delete_template(self, name: str):
        """Delete a template file and remove it from memory"""
        template = self.templates.get(name)
        if not template:
            raise KeyError(f"Template not found: {name}")

        if template.author.lower() == 'system':
            raise ValueError("Cannot delete system templates")

        # Remove from memory
        del self.templates[name]

        # Delete file from user templates directory
        template_file = self.user_template_dir / f"{name}.yaml"
        if template_file.exists():
            template_file.unlink()
            logging.info(f"Deleted template file: {template_file}")

    def template_exists(self, name: str, is_final: bool = False) -> bool:
        """Check if a template exists by name"""
        return name in (self.final_templates if is_final else self.templates)

# Update the existing PromptTemplates class to use TemplateManager
class PromptTemplates:
    def __init__(self, template_path: Optional[str] = None):
        self.manager = TemplateManager(template_path or "templates")

    def get_template(self, template_name: str, is_final: bool = False) -> Optional[str]:
        try:
            return self.manager.get_template(template_name, is_final)
        except KeyError:
            logging.error(f"Template not found: {template_name}")
            return None
        except Exception as e:
            logging.error(f"Error retrieving template {template_name}: {e}")
            return None

    def get_templates_with_descriptions(self) -> Dict[str, Tuple[str, str]]:
        """Get regular analysis templates with descriptions"""
        templates = self.manager.list_templates(include_final=False)
        return {
            name: (
                self.manager.get_template(name),
                data['description']
            )
            for name, data in templates.items()
        }

    def get_final_templates_with_descriptions(self) -> Dict[str, Tuple[str, str]]:
        """Get final analysis templates with descriptions"""
        templates = self.manager.list_final_templates()
        return {
            name: (
                self.manager.get_template(name, is_final=True),
                data['description']
            )
            for name, data in templates.items()
        }