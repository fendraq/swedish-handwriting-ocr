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

def add_footer_filename(c, filename, pdf_config):
    """
    Helper function to add template filename to footer (centered at bottom).
    
    Args:
        c: ReportLab canvas object
        filename: Template filename to display
        pdf_config: PDF configuration dictionary
    """
    footer_text = f"Template: {filename}"
    footer_font_size = 8
    c.setFont("Helvetica", footer_font_size)
    c.setFillColorRGB(0.5, 0.5, 0.5)  # Gray color
    
    # Position footer at bottom center
    footer_width = c.stringWidth(footer_text, "Helvetica", footer_font_size)
    footer_x = (pdf_config['page_width'] - footer_width) / 2
    footer_y = 15 * mm  # 15mm from bottom
    c.drawString(footer_x, footer_y, footer_text)
    
    # Reset to default color and font
    c.setFillColorRGB(0, 0, 0)  # Back to black
    c.setFont("Helvetica", pdf_config['font_size'])

def create_template_pdf(all_items, output_dir, pdf_config, format_type='single-rows', output_name='template'):
    """ Create a template PDF with specified format """
    from datetime import datetime
    
    # Generate unique timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Map format_type to short suffix for filename (matches orchestrator expectations)
    format_suffix = 'sl' if format_type == 'single-rows' else 'tf'
    base_name = f"swedish_handwriting_{format_suffix}_{timestamp}"
    
    # Path for PDF
    filename = f"{output_dir}/{base_name}.pdf"
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

        # Handle regular lines
        text = item['text']
        category = item['category']
        layout = item['layout']

        # Update layout if changed
        if current_layout != layout:
            current_layout = layout
            if current_col > 0:
                current_col = 0
                current_row += 1

        # Check if this is text-field format
        if format_type == 'text-field':
            # TEXT-FIELD FORMAT: Large text area per page
            
            # Start new page for each note
            if len(metadata) > 0:  # Not the first item
                # Add footer before showing next page
                add_footer_filename(c, f"{base_name}.pdf", pdf_config)
                c.showPage()
                page_number += 1
                marker_gen.draw_markers(c)
                add_page_number(c, page_number, pdf_config)
            
            # Text area positioning (upper half of page)
            text_start_y = pdf_config['page_height'] - pdf_config['margin'] - 20*mm
            text_area_height = 80*mm  # Upper portion for text
            
            # Draw the reference text with word wrapping
            c.setFont('Helvetica', pdf_config['font_size'])
            
            # Simple word wrapping
            words = text.split()
            lines = []
            current_line = ""
            line_width = pdf_config['usable_width'] - 20*mm
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if c.stringWidth(test_line, 'Helvetica', pdf_config['font_size']) <= line_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Draw text lines
            text_y = text_start_y
            line_height = pdf_config['font_size'] * 1.4
            for line in lines:
                c.drawString(pdf_config['margin'] + 10*mm, text_y, line)
                text_y -= line_height
            
            # Large handwriting area (lower half)
            rect_x = pdf_config['margin']
            rect_y = pdf_config['margin'] + 20*mm
            rect_width = pdf_config['usable_width']
            rect_height = layout['text_area_height']
            
            c.setLineWidth(1)
            c.rect(rect_x, rect_y, rect_width, rect_height, stroke=1, fill=0)
            
            # Store metadata for segmentation (TEXT-FIELD)
            note_metadata = {
                'text': text,
                'page': page_number,
                'position': [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],  # For segmentation compatibility
                'handwriting_position': [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                'reference_text_position': [pdf_config['margin'] + 10*mm, text_start_y - len(lines)*line_height, 
                                          pdf_config['margin'] + 10*mm + line_width, text_start_y],
                'category': category,
                'subcategory': item.get('subcategory', None),
                'word_id': f"{len(metadata):03d}"  # For segmentation compatibility
            }
            metadata.append(note_metadata)
            
        else:
            # LINE-LEVEL FORMAT: Original behavior
            items_per_row = layout['items_per_row']
            item_width = layout['item_width']
            row_height = layout['row_height']
            max_rows = layout['max_rows']

            # Check if new page is needed
            if current_row >= max_rows:
                # Add footer before showing next page
                add_footer_filename(c, f"{base_name}.pdf", pdf_config)
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

            # Draw reference text (printed text to copy)
            c.setFont('Helvetica', pdf_config['font_size'])
            c.drawString(x, y, text)
            
            # Draw handwriting area below reference text
            rect_x = x
            rect_y = y - pdf_config['handwriting_space']
            rect_width = item_width - 6*mm
            rect_height = pdf_config['handwriting_space'] - 2*mm
            c.setLineWidth(0.5)
            c.rect(rect_x, rect_y, rect_width, rect_height, stroke=1, fill=0)
            c.setLineWidth(1)
            
            # Store metadata for segmentation (SINGLE-ROWS)
            line_metadata = {
                'text': text,
                'page': page_number,
                'position': [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],  # For segmentation compatibility
                'handwriting_position': [rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                'reference_text_position': [x, y, x + rect_width, y + (pdf_config['font_size'] * 1.2)],
                'category': category,
                'subcategory': item.get('subcategory', None),
                'word_id': f"{len(metadata):03d}"  # For segmentation compatibility
            }
            metadata.append(line_metadata)

            # Move to current position
            current_col += 1
            if current_col >= items_per_row:
                current_col = 0
                current_row += 1

    # Add footer to final page
    add_footer_filename(c, f"{base_name}.pdf", pdf_config)
    
    # Save PDF after all items are processed
    c.save()
    print(f"Complete PDF created: {filename}")

    # Save metadata for segmentation
    metadata_file = f"{output_dir}/{base_name}.json"

    # Calculate relative positions for coordinate transformation
    all_coords = [item['position'] for item in metadata]  # Use 'position' for compatibility
    relative_coords = marker_gen.calculate_relative_coordinates(all_coords)

    for i, item in enumerate(metadata):
        item['relative_position'] = relative_coords[i]  # Standard field for segmentation

    # Determine data structure based on format_type parameter
    # KEEP OLD STRUCTURE FOR SEGMENTATION COMPATIBILITY
    total_key = 'total_words'  # Always use 'words' for segmentation compatibility
    items_key = 'words'        # Always use 'words' for segmentation compatibility
    total_count = len(metadata)

    # For line-level: change line_id to word_id for compatibility
    if format_type == 'single-rows':
        for item in metadata:
            if 'line_id' in item:
                item['word_id'] = item['line_id']  # Rename for compatibility
                del item['line_id']
    
    # For text-field: change note_id to word_id for compatibility  
    if format_type == 'text-field':
        for item in metadata:
            if 'note_id' in item:
                item['word_id'] = item['note_id']  # Rename for compatibility
                del item['note_id']

    metadata_structure = {
        'pdf_file': filename,
        total_key: total_count,
        'total_pages': page_number,
        'categories': list(set(item['category'] for item in metadata)),
        'reference_system': marker_gen.get_marker_metadata(),
        'format_type': format_type,
        items_key: metadata
    }
        
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_structure, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved: {metadata_file}")
    print(f"Reference markers added to all {page_number} pages")
    print(f"Format: {format_type.upper()}")
    return filename, metadata_file


def test_basic_pdf_generation(output_dir):
    """Test basic PDF creation with ReportLab."""
    # Enkel test för att se att ReportLab fungerar
    test_file = f"{output_dir}/test.pdf"
    c = canvas.Canvas(test_file, pagesize=A4)
    c.drawString(100, 750, "Test PDF - Swedish Handwriting Templates")
    c.save()
    print(f"Test PDF created: {test_file}")