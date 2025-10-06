"""
PDF generation using ReportLab for handwriting templates.
"""
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from .reference_markers import ReferenceMarkerGenerator

def add_page_number(c, page_number, pdf_config):
    """
    Helper function to add page number to current page.
    
    Args:
        c: ReportLab canvas object
        page_number: Current page number
        pdf_config: PDF configuration dictionary
    """
    page_num_text = f"Sida {page_number}"
    page_num_y = pdf_config['margin'] / 2
    page_num_x = pdf_config['page_width'] - pdf_config['margin'] - c.stringWidth(page_num_text, "Helvetica", 10)
    c.setFont("Helvetica", 10)
    c.drawString(page_num_x, page_num_y, page_num_text)
    c.setFont("Helvetica", pdf_config['font_size'])

def create_template_pdf(all_items, output_dir, pdf_config):
    """ Create a template for PDF with all categories, each category as section """
    # Path for PDF
    filename = f"{output_dir}/swedish_handwriting_template.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    marker_gen = ReferenceMarkerGenerator(
        pdf_config['page_width'],
        pdf_config['page_height'],
        pdf_config['margin']
    )

    # Validate marker positions
    warnings = marker_gen.validate_marker_positions()
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")

    # Position tracking
    current_row = 0
    current_col = 0
    page_number = 1
    metadata= []
    current_layout = None
    first_page_markers_drawn = False

    for item in all_items:
        if not first_page_markers_drawn:
            marker_gen.draw_markers(c)

            # Add page number at bottom of page
            add_page_number(c, page_number, pdf_config)

            instruction_text = "Skanna in i jpg-format och skriv innanför linjerna"
            instruction_y = pdf_config['page_height'] - (pdf_config['margin'] / 2)
            c.setFont("Helvetica", 10)

            # Calculate text width and center it
            text_width = c.stringWidth(instruction_text, "Helvetica", 10)
            page_width = pdf_config['page_width']
            x_centered = (page_width - text_width) / 2

            c.drawString(x_centered, instruction_y, instruction_text)
            c.setFont("Helvetica", pdf_config['font_size'])  # Reset font

            first_page_markers_drawn = True

        if item.get('is_header', False):
            if current_row > 0:
                c.showPage()
                page_number += 1
                current_row = 0
                current_col = 0

                # Add page number to new page after header-driven page break
                add_page_number(c, page_number, pdf_config)

            marker_gen.draw_markers(c)

            header_y = pdf_config['page_height'] - pdf_config['margin']
            c.setFont("Helvetica-Bold", 16)
            c.drawString(pdf_config['margin'], header_y, item['text'])
            c.setFont("Helvetica", pdf_config['font_size'])
            current_row = 2
            continue

        # Handle regular words
        text = item['text']
        category = item['category']
        layout = item['layout']

        # Update laout if changed
        if current_layout != layout:
            current_layout = layout
            if current_col > 0:
                current_col = 0
                current_row += 1

        items_per_row = layout['items_per_row']
        item_width = layout['item_width']
        row_height = layout['row_height']
        max_rows = layout['max_rows']

        # Check if new page is needed
        if current_row >= max_rows:
            c.showPage()
            page_number += 1
            current_row = 0
            current_col = 0
            first_page_markers_drawn = False

            marker_gen.draw_markers(c)
            # Add page number to new page after max_rows page break
            add_page_number(c, page_number, pdf_config)

        # Calculate position
        x = pdf_config['margin'] + (current_col * item_width)
        y = pdf_config['page_height'] - pdf_config['margin'] - (current_row * row_height)

        # Draw reference area
        c.setFont('Helvetica', pdf_config['font_size'])
        c.drawString(x, y, text)

        # Draw handwriting area
        rect_x = x
        rect_y = y - pdf_config['handwriting_space']
        rect_width = item_width - 6*mm
        rect_height = pdf_config['handwriting_space'] - 2*mm
        c.setLineWidth(0.5)
        c.rect(rect_x, rect_y, rect_width, rect_height, stroke=1, fill=0)
        c.setLineWidth(1)

        # Store metadata for segmentation
        word_metadata = {
            'text': text,
            'page': page_number, 
            'position': [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height], 
            'category': category,
            'subcategory': item.get('subcategory', None),
            'word_id': f"{category}_{len(metadata):03d}"
        }
        metadata.append(word_metadata)

        # Move to current position
        current_col += 1
        if current_col >= items_per_row:
            current_col = 0
            current_row += 1

    # Save PDF after all items are processed
    c.save()
    print(f"Complete PDF created: {filename}")

    # Save metadata for segmentation
    metadata_file = f"{output_dir}/complete_template_metadata.json"

    all_words_coords = [word['position'] for word in metadata]
    relative_coords = marker_gen.calculate_relative_coordinates(all_words_coords)

    for i, word in enumerate(metadata):
        word['relative_position'] = relative_coords[i]

    metadata_structure = {
        'pdf_file': filename,
        'total_words': len(metadata),
        'total_pages': page_number,
        'categories': list(set(item['category'] for item in metadata)),
        'reference_system': marker_gen.get_marker_metadata(),
        'words': metadata
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_structure, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved: {metadata_file}")
    print(f"Reference markers added to all {page_number} pages")
    return filename, metadata_file


def test_basic_pdf_generation(output_dir):
    """Test basic PDF creation with ReportLab."""
    # Enkel test för att se att ReportLab fungerar
    test_file = f"{output_dir}/test.pdf"
    c = canvas.Canvas(test_file, pagesize=A4)
    c.drawString(100, 750, "Test PDF - Swedish Handwriting Templates")
    c.save()
    print(f"Test PDF created: {test_file}")