import json

def analyze_json_structure(json_file_path):
    """Analyze the JSON structure to understand the data."""
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

        # Count items
        if 'items' in category_data:
            item_count = len(category_data['items'])
            print(f" Items: {item_count}")
        elif 'subcategories' in category_data:
            total_items = sum(len(subcat) for subcat in category_data['subcategories'].values())
            print(f" Total items in subcategories: {total_items}")

    return data


def load_json(file_path):
    """ Load and return JSON data """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_items_from_category(category_data):
    """ Extract all file items from category, handling different structures. """
    items = []
    
    if 'items' in category_data:
        # Direct items list
        for item in category_data['items']:
            if 'variants' in item:
                # Word with variants
                for variant in item['variants']:
                    items.append({'text': variant, 'subcategory': None})
            elif 'text' in item:
                # Sentence or complex text
                items.append({'text': item['text'], 'subcategory': None})
    
    elif 'subcategories' in category_data:
        # Handle subcategories
        for subcat_name, subcat_items in category_data['subcategories'].items():
            if isinstance(subcat_items, list):
                if len(subcat_items) > 0 and isinstance(subcat_items[0], dict):
                    # List of objects with variants
                    for item in subcat_items:
                        # Add subcategory info to each item
                        for variant in item['variants']:
                            items.append({'text': variant, 'subcategory': subcat_name})
                else:
                    # Simple string list
                    for text_item in subcat_items:
                        items.append({'text': text_item, 'subcategory': subcat_name})
    
    return items