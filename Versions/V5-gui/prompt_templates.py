from pathlib import Path
import yaml
from typing import Dict, Optional, Tuple

class PromptTemplates:
    def __init__(self, template_path: Optional[str] = None):
        self.templates = self._load_default_templates()
        self.template_path = template_path
        if template_path:
            self.load_templates(template_path)

    def _load_default_templates(self) -> Dict[str, Tuple[str, str]]:
        return {
            'analysis': ("""
                For each file:
                1. List all imports and external dependencies
                2. Summarize the main purpose of the file
                3. List key functions and classes
                4. Identify any potential issues or improvements
            """.strip(),
            "General code analysis with dependencies and structure"),
            
            'requirements': ("""
                For each file:
                1. List all imports and external dependencies
            """.strip(),
            "Extract dependencies and import requirements"),
            
            'security_analysis': ("""
                Analyze the code for security concerns:
                1. Identify potential security vulnerabilities
                2. Check for hardcoded credentials
                3. Review authentication mechanisms
                4. Assess data validation
                5. Check for secure communication
            """.strip(),
            "Security-focused code review and vulnerability assessment"),
            
            'performance_analysis': ("""
                Review code for performance optimization:
                1. Identify performance bottlenecks
                2. Check for inefficient algorithms
                3. Review database queries
                4. Analyze memory usage
                5. Look for unnecessary computations
            """.strip(),
            "Performance optimization and bottleneck detection"),
            
            'final_analysis': ("""
                Analyze all the code findings and provide:
                1. Overall architecture overview
                2. Common patterns and practices
                3. Key improvement recommendations
                4. Technical debt assessment
                5. Project health summary
            """.strip(),
            "Final summary and architectural overview")
        }

    def load_templates(self, path: str):
        """Load templates from a YAML file"""
        template_path = Path(path)
        if template_path.exists():
            with open(template_path, 'r') as f:
                custom_templates = yaml.safe_load(f)
                self.templates.update(custom_templates)

    def save_templates(self, path: str):
        """Save templates to a YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.templates, f)

    def add_template(self, name: str, content: str, description: str = ""):
        """Add a new template with optional description"""
        self.templates[name] = (content, description)

    def remove_template(self, name: str):
        """Remove a template"""
        if name in self.templates:
            del self.templates[name]

    def get_template(self, template_name: str) -> str:
        """Get a template content by name"""
        template = self.templates.get(template_name)
        return template[0] if isinstance(template, tuple) else template

    def get_templates_with_descriptions(self) -> Dict[str, Tuple[str, str]]:
        """Get all templates with their descriptions"""
        return {name: (
            template[0] if isinstance(template, tuple) else template,
            template[1] if isinstance(template, tuple) else "No description"
        ) for name, template in self.templates.items()}