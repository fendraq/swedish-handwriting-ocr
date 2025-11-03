"""
Test line generation for all writers with uppercase and lowercase
"""
import logging
from pathlib import Path
from scripts.data_processing.synthetic_data.line_generator import LineGenerator
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*60)
    print("TESTING ALL WRITERS - UPPERCASE & LOWERCASE")
    print("="*60 + "\n")
    
    # Initialize generator
    generator = LineGenerator()
    
    # Load words
    words_by_writer = generator.load_word_images()
    
    test_output_dir = Path('dataset/trocr_ready_data/v2/test_all_writers')
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_tests = 0
    successful_tests = 0
    
    # Test each writer
    for writer_id in sorted(words_by_writer.keys()):
        words = words_by_writer[writer_id]
        
        # Group by case
        uppercase = [w for w in words if w[1].isupper()]
        lowercase = [w for w in words if w[1].islower()]
        mixed = [w for w in words if not w[1].isupper() and not w[1].islower()]
        
        print(f"\n{writer_id}:")
        print(f"  Uppercase: {len(uppercase)}, Lowercase: {len(lowercase)}, Mixed: {len(mixed)}")
        
        # Test uppercase line
        if len(uppercase) >= 5:
            total_tests += 1
            try:
                import random
                selected = random.sample(uppercase, 5)
                cropped = [generator.crop_word(p) for p, _ in selected]
                texts = [t for _, t in selected]
                
                line_img, line_text = generator.combine_words_to_line(cropped, texts)
                
                output_file = test_output_dir / f"{writer_id}_UPPERCASE.jpg"
                cv2.imwrite(str(output_file), line_img)
                
                print(f"  ✓ UPPERCASE: '{line_text}' ({line_img.shape[1]}px)")
                successful_tests += 1
            except Exception as e:
                print(f"  ✗ UPPERCASE failed: {e}")
        else:
            print(f"  ⊘ UPPERCASE: Not enough words ({len(uppercase)})")
        
        # Test lowercase line
        if len(lowercase) >= 5:
            total_tests += 1
            try:
                import random
                selected = random.sample(lowercase, 5)
                cropped = [generator.crop_word(p) for p, _ in selected]
                texts = [t for _, t in selected]
                
                line_img, line_text = generator.combine_words_to_line(cropped, texts)
                
                output_file = test_output_dir / f"{writer_id}_lowercase.jpg"
                cv2.imwrite(str(output_file), line_img)
                
                print(f"  ✓ lowercase: '{line_text}' ({line_img.shape[1]}px)")
                successful_tests += 1
            except Exception as e:
                print(f"  ✗ lowercase failed: {e}")
        else:
            print(f"  ⊘ lowercase: Not enough words ({len(lowercase)})")
    
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {successful_tests}/{total_tests} successful")
    print("="*60)
    print(f"\nTest images saved to: {test_output_dir}/")
    print("Review images to verify:")
    print("  1. Cropping is clean (no clipped text)")
    print("  2. Same height for tallest word per line")
    print("  3. Natural spacing between words")
    print("  4. Bottom-aligned baseline")
    print("\n")

if __name__ == '__main__':
    main()
