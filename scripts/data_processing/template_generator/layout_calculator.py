from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def setup_pdf_config(template_config):
    """ Setup PDF configurations from JSON. """
    # Page dimensions
    width, height = A4
    
    # Parse margin (remove 'mm' suffix and convert to float)
    margin_str = template_config['margin'].replace('mm', '')
    margin = float(margin_str) * mm
    
    font_size = template_config['font_size']
    
    # Parse handwriting_space or text_area_height
    if 'handwriting_space' in template_config:
        handwriting_space_str = template_config['handwriting_space'].replace('mm', '')
        handwriting_space = float(handwriting_space_str) * mm
        text_area_height = None
    else:
        # text-field format
        text_area_height_str = template_config['text_area_height'].replace('mm', '')
        text_area_height = float(text_area_height_str) * mm
        handwriting_space = None

    # Calculate usable area
    usable_width = width - (2 * margin)
    usable_height = height - (2 * margin)

    config = {
        'page_width': width,
        'page_height': height,
        'margin': margin,
        'usable_width': usable_width,
        'usable_height': usable_height,
        'font_size': font_size
    }
    
    if handwriting_space:
        config['handwriting_space'] = handwriting_space
    if text_area_height:
        config['text_area_height'] = text_area_height

    return config


def calculate_layout(category_data, pdf_config, format_type='single-rows'):
    """ 
    Calculate layout parameters based on category settings.
    Supports both line-level (1 line per row) and text-field (1 note per page) layouts.
    """
    layout_type = category_data['template_layout']

    # Check if this is text-field format
    is_text_field = (format_type == 'text-field')
    
    if is_text_field:
        # Text-field format: 1 note per page with large text area
        layout = {
            'items_per_row': 1,
            'item_width': pdf_config['usable_width'],
            'items_per_page': 1,
            'max_rows': 1,
            'text_field_format': True,
            'text_area_height': pdf_config['text_area_height']
        }
    else:
        # Line-level format: multiple lines per page
        # Parse layout type (e.g., "1_line_per_row" -> 1)
        if 'per_row' in layout_type:
            items_per_row = int(layout_type.split('_')[0])
        else:
            items_per_row = 1
        
        # Calculate spacing - full width for lines
        item_width = pdf_config['usable_width'] / items_per_row

        # Row height calculation - slightly tighter for more lines per page
        row_height = pdf_config['handwriting_space'] + (pdf_config['font_size'] * 1.2)
        
        max_rows = int(pdf_config['usable_height'] / row_height) - 1

        layout = {
            'items_per_row': items_per_row,
            'item_width': item_width,
            'row_height': row_height,
            'max_rows': max_rows,
            'items_per_page': items_per_row * max_rows,
            'text_field_format': False
        }

    return layout