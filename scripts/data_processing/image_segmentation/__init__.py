from .coordinate_converter import get_words_for_page, load_metadata, pdf_to_pixel_coordinates, validate_coordinates
from .image_processor import load_scanned_image, extract_word_region, enhance_word_image
from .file_manager import create_output_structure, save_word_segment, generate_segmentation_report

from pathlib import Path
from typing import Dict, List

class ImageSegmenter:
    """
    Main Class which coordinate segmentation process
    """
    
    def __init__(self, metadata_path: str, output_dir: str):
        """
        Initiate the segmentation
        
        Args:
            metadata_path: Path to complete_template_metadata.json
            output_dir: Folder for segmented images
        """

        self.metadata = load_metadata(metadata_path)

        self.categories = list(self.metadata['categories'])

        self.output_dir = output_dir

        self.output_paths = None

        print(f"ImageSegmenter initialized:")
        print(f"  Total words in metadata: {self.metadata['total_words']}")
        print(f"  Categories: {self.categories}")
    
    def segment_single_page(self, image_path: str, page_number: int, writer_id: str) -> List[str]:
        """
        Segmenterar en enda skannad sida.
        
        Args:
            image_path: Sökväg till skannad bild
            page_number: Vilken sida (1, 2, 3...)
            writer_id: Skribent-ID
            
        Returns:
            Lista med sökvägar till segmenterade ordbilder
        """
        # 1. Ladda bild
        image = load_scanned_image(image_path)
        img_height, img_width = image.shape[:2]
        
        # 2. Få ord för denna sida
        page_words = get_words_for_page(self.metadata, page_number)
        
        segmented_files = []
        
        # 3. För varje ord på sidan
        for word_data in page_words:
            try:
                # Konvertera PDF-koordinater till pixlar
                pdf_coords = word_data['position']
                pixel_coords = pdf_to_pixel_coordinates(pdf_coords, img_width, img_height)
                
                # Validera koordinater
                safe_coords = validate_coordinates(pixel_coords, image.shape)
                
                # Extrahera ordregion
                word_region = extract_word_region(image, safe_coords)
                
                # Förbättra bild (valfritt)
                enhanced_word = enhance_word_image(word_region)
                
                # Spara ordbilden
                saved_path = save_word_segment(
                    enhanced_word,
                    word_data['text'],
                    word_data['category'],
                    writer_id,
                    word_data['word_id'],
                    self.output_paths
                )
                
                segmented_files.append(saved_path)
                
            except Exception as e:
                print(f"Warning: Failed to segment word '{word_data['text']}': {e}")
                continue
        
        print(f"Segmented {len(segmented_files)} words from page {page_number}")
        return segmented_files
    
    def segment_multiple_pages(self, images_dir: str, writer_id: str = "writer_001") -> Dict[str, List[str]]:
        """
        Segmenterar alla bilder i en mapp.
        
        Args:
            images_dir: Mapp med skannade bilder
            writer_id: Skribent-ID
            
        Returns:
            Dictionary med source_image -> [segmented_files]
        """
        # 1. Skapa output-struktur
        self.output_paths = create_output_structure(self.output_dir, writer_id, self.categories)
        
        # 2. Hitta alla JPG-filer
        images_path = Path(images_dir)
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.JPG"))
        
        if not image_files:
            print(f"No JPG images found in {images_dir}")
            return {}
        
        # Sortera för förutsägbar ordning
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} images to process")
        
        results = {}
        
        # 3. För varje bild
        for i, image_file in enumerate(image_files):
            page_number = i + 1  # Första bilden = sida 1
            print(f"\nProcessing page {page_number}: {image_file.name}")
            
            try:
                segmented_files = self.segment_single_page(str(image_file), page_number, writer_id)
                results[str(image_file)] = segmented_files
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results[str(image_file)] = []
        
        # 5. Generera rapport
        generate_segmentation_report(results, self.output_dir)
        
        return results