#!/usr/bin/env python3
"""
Main CLI script for generating Swedish handwriting templates.
"""
from pathlib import Path
from config.paths import DocsPaths
from .template_generator import TemplateGenerator
from .template_generator.json_parser import analyze_json_structure

def main():
    # Paths
    json_file = str(DocsPaths.WORD_COLLECTIONS / "swedish_words_funeral.json")
    output_dir = str(DocsPaths.GENERATED_TEMPLATES)  

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Start with analysis
    print("Analyzing JSON structure...")
    analyze_json_structure(json_file)
    
    
    # Create generator
    generator = TemplateGenerator(json_file, output_dir)
    pdf_file, metadata_file = generator.generate_pdf()
    

    print(f"\nGeneration complete")
    print(f"PDF: {pdf_file}")
    print(f"Metadata: {metadata_file}")

if __name__ == "__main__":
    main()