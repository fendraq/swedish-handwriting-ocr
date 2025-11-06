import json

def analyze_json_structure(json_file_path):
    """Analyze the JSON structure to understand the data (line-level only)."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Basic info
    print(f"Dataset: {data['dataset_info']['name']}")
    print(f"Version: {data['dataset_info']['version']}")
    print(f"Categories: {len(data['categories'])}")

    # Analyze each category
    for category_name, category_data in data['categories'].items():
        print(f"\n{category_name}:")
        print(f" Description: {category_data['description']}")
        print(f" Layout: {category_data['template_layout']}")
        print(f" Difficulty: {category_data['difficulty']}")
        
        # Count lines or notes
        if 'lines' in category_data:
            line_count = len(category_data['lines'])
            print(f" Lines: {line_count}")
        elif 'notes' in category_data:
            note_count = len(category_data['notes'])
            print(f" Notes: {note_count}")

    return data


def load_json(file_path):
    """ Load and return JSON data """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_items_from_category(category_data, format_type='single-rows'):
    """ 
    Extract items from category based on format type.
    
    Args:
        category_data: Category data from JSON
        format_type: 'single-rows' or 'text-field'
    
    Returns:
        list: Items with 'text' and 'subcategory' fields
    """
    items = []
    
    if format_type == 'text-field':
        # Extract notes for text-field format
        for note_text in category_data['notes']:
            items.append({'text': note_text, 'subcategory': None})
    else:
        # Extract lines for single-rows format
        for line_text in category_data['lines']:
            items.append({'text': line_text, 'subcategory': None})
    
    return items