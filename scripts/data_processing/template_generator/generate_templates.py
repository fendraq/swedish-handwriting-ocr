#!/usr/bin/env python3
"""
Main CLI script for generating Swedish handwriting templates.
"""
import argparse
from pathlib import Path
from config.paths import DocsPaths
from . import TemplateGenerator
from .json_parser import analyze_json_structure

def main():
    parser = argparse.ArgumentParser(
        description="Generate Swedish handwriting templates (PDF + metadata)"
    )
    
    parser.add_argument(
        "--type",
        choices=["lines", "text-field"],
        default="lines",
        help="Type of template to generate: 'lines' for single-line or 'text-field' for text field templates (default: lines)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input JSON file. If not specified, uses default based on type."
    )
    
    args = parser.parse_args()
    
    # Determine input file based on type if not specified
    if args.input:
        json_file = args.input
    else:
        if args.type == "lines":
            json_file = str(DocsPaths.LINE_TEXTS / "swedish_sentences_v2.json")
        else:  # text-field
            json_file = str(DocsPaths.WORD_COLLECTIONS / "swedish_words_funeral.json")
    
    # Validate input file exists
    if not Path(json_file).exists():
        print(f"Error: Input file not found: {json_file}")
        return
    
    output_dir = str(DocsPaths.GENERATED_TEMPLATES)  

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Start with analysis
    print(f"Template type: {args.type}")
    print(f"Input file: {json_file}")
    print("\nAnalyzing JSON structure...")
    analyze_json_structure(json_file)
    
    
    # Create generator
    generator = TemplateGenerator(json_file, output_dir)
    pdf_file, metadata_file = generator.generate_pdf()
    

    print(f"\nGeneration complete")
    print(f"PDF: {pdf_file}")
    print(f"Metadata: {metadata_file}")

if __name__ == "__main__":
    main()