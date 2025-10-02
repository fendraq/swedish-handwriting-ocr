from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

def setup_pdf_config(template_config):
    """ Setup PDF configurations from JSON. """
    page_settings = template_config['page_settings']

    # Page dimensions
    width, height = A4
    margin = page_settings['margin_mm'] * mm
    font_size = page_settings['font_size_pt']
    handwriting_space = page_settings['handwriting_space_mm'] * mm

    # Calculate usable area
    usable_width = width - (2 * margin)
    usable_height = height - (2 * margin)

    config = {
        'page_width': width,
        'page_height': height,
        'margin': margin,
        'usable_width': usable_width,
        'usable_height': usable_height,
        'font_size': font_size,
        'handwriting_space': handwriting_space
    }

    return config


def calculate_layout(category_data, pdf_config):
    """ Calculate layout parameters based on category settings. """
    layout_type = category_data['template_layout']

    # Parse layout type (e.g., "3_words_per_row" -> 3)
    if 'per_row' in layout_type:
        items_per_row = int(layout_type.split('_')[0])
    else:
        items_per_row = 1

    # Calculate spacing
    item_width = pdf_config['usable_width'] / items_per_row

    # Estimate rows per page
    row_height = pdf_config['handwriting_space'] + (pdf_config['font_size'] *2)
    max_rows = int(pdf_config['usable_height'] / row_height)

    layout = {
        'items_per_row': items_per_row,
        'item_width': item_width,
        'row_height': row_height,
        'max_rows': max_rows,
        'items_per_page': items_per_row * max_rows
    }

    return layout