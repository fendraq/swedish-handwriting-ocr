"""
Template Generator Package for Swedish Handwriting Dataset.
"""
from pathlib import Path
from .json_parser import load_json, extract_items_from_category
from .layout_calculator import setup_pdf_config, calculate_layout
from .pdf_generator import create_template_pdf

class TemplateGenerator:
    """Main orchestrator class that coordinates all template generation."""
    
    def __init__(self, json_file_path, output_dir, format_type='single-rows', output_name=None):
        self.data = load_json(json_file_path)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        self.output_name = output_name or format_type
        self.pdf_config = setup_pdf_config(self.data['template_config'])
        self.metadata = {}  # Store positions for segmentation
    
    def generate_pdf(self):
        """Generate PDF with specified format"""
        all_items_with_categories = []
        
        for category_name, category_data in self.data['categories'].items():
            # Get layout for category
            layout = calculate_layout(category_data, self.pdf_config, self.format_type)

            # Add items in category
            items = extract_items_from_category(category_data, self.format_type)
            for item in items:
                item['category'] = category_name
                item['layout'] = layout
                all_items_with_categories.append(item)

        return create_template_pdf(all_items_with_categories, self.output_dir, self.pdf_config, 
                                 self.format_type, self.output_name)
        
