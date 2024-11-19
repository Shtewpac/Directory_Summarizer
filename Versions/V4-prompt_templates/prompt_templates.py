
from pathlib import Path
import yaml
from typing import Dict, Optional

class PromptTemplates:
    def __init__(self, template_path: Optional[str] = None):
        self.templates = self._load_default_templates()
        if template_path:
            self.load_templates(template_path)

    def _load_default_templates(self) -> Dict[str, str]:
        return {
            'analysis': """
                For each file:
                1. List all imports and external dependencies
                2. Summarize the main purpose of the file
                3. List key functions and classes
                4. Identify any potential issues or improvements
            """.strip(),
            'requirements': """
                For each file:
                1. List all imports and external dependencies
            """.strip(),
            'final_analysis': """
                Analyze all the code findings and provide:
                1. Overall architecture overview
                2. Common patterns and practices
                3. Key improvement recommendations
                4. Technical debt assessment
                5. Project health summary
            """.strip()
        }

    def load_templates(self, path: str):
        """Load templates from a YAML file"""
        template_path = Path(path)
        if template_path.exists():
            with open(template_path, 'r') as f:
                custom_templates = yaml.safe_load(f)
                self.templates.update(custom_templates)

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        return self.templates.get(template_name, '')