"""
Template Generator Package for Swedish Handwriting Dataset.
"""
from pathlib import Path
from .json_parser import load_json, extract_items_from_category
from .layout_calculator import setup_pdf_config, calculate_layout
from .pdf_generator import create_pdf_for_category

class TemplateGenerator:
    """Main orchestrator class that coordinates all template generation."""
    
    def __init__(self, json_file_path, output_dir):
        self.data = load_json(json_file_path)
        self.output_dir = Path(output_dir)
        self.pdf_config = setup_pdf_config(self.data['template_config'])
        self.metadata = {}  # Store word positions for segmentation
    
    def generate_category_templates(self, category_name):
        """Generate templates for a specific category."""
        category_data = self.data['categories'][category_name]
        layout = calculate_layout(category_data, self.pdf_config)
        
        # Extract items based on category structure
        items = extract_items_from_category(category_data)
        
        # Generate PDF pages
        create_pdf_for_category(category_name, items, layout, self.output_dir, self.pdf_config)