"""
PDF generation using ReportLab for handwriting templates.
"""
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def create_template_pdf(all_items, output_dir, pdf_config):
    """ Create a template for PDF with all categories, each category as section """
    # Path for PDF
    filename = f"{output_dir}/swedish_handwriting_template.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    # Position tracking
    current_row = 0
    current_col = 0
    page_number = 1
    metadata= []
    current_layout = None

    for item in all_items:
        if item.get('is_header', False):
            if current_row > 0:
                c.showPage()
                page_number += 1
                current_row = 0
                current_col = 0

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

        # Calculate position
        x = pdf_config['margin'] + (current_col * item_width)
        y = pdf_config['page_height'] - pdf_config['margin'] - (current_row * row_height)

        # Draw reference area
        c.drawString(x, y, text)

        # Draw handwriting area
        rect_x = x
        rect_y = y - pdf_config['handwriting_space']
        rect_width = item_width - 10*mm
        rect_height = pdf_config['handwriting_space'] - 5*mm
        c.rect(rect_x, rect_y, rect_width, rect_height, stroke=1, fill=0)

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
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'pdf_file': filename,
            'total_words': len(metadata),
            'total_pages': page_number,
            'categories': list(set(item['category'] for item in metadata)),
            'words': metadata
        }, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved: {metadata_file}")
    return filename, metadata_file


def test_basic_pdf_generation(output_dir):
    """Test basic PDF creation with ReportLab."""
    # Enkel test för att se att ReportLab fungerar
    test_file = f"{output_dir}/test.pdf"
    c = canvas.Canvas(test_file, pagesize=A4)
    c.drawString(100, 750, "Test PDF - Swedish Handwriting Templates")
    c.save()
    print(f"Test PDF created: {test_file}")