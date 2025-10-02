#!/usr/bin/env python3
"""
Main CLI script for generating Swedish handwriting templates.
"""
from pathlib import Path
from template_generator import TemplateGenerator
from template_generator.json_parser import analyze_json_structure

def main():
    # Paths
    json_file = "../../dataset/swedish_words.json"
    output_dir = "../../dataset/templates/generated_templates"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Start with analysis
    print("Analyzing JSON structure...")
    analyze_json_structure(json_file)
    
    # Test basic PDF generation
    from template_generator.pdf_generator import test_basic_pdf_generation
    test_basic_pdf_generation(output_dir)
    
    # Create generator
    generator = TemplateGenerator(json_file, output_dir)
    
    # Test with one category first
    generator.generate_category_templates('basic_words')

if __name__ == "__main__":
    main()