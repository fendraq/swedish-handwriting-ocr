import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from config.paths import ensure_dir

word_path = Path(__file__).parent / "synthetic_words.json"
img_size = 384
font_path = "config/fonts/ReenieBeanie-Regular.ttf"  
max_text_height = 0.155
max_text_size = int(img_size * max_text_height)

def load_words():
    with open(word_path, encoding="utf-8") as f:
        return json.load(f)

def get_font(font_size):
    return ImageFont.truetype(font_path, font_size)

def render_word(word, font):
    img = Image.new("L", (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_size - w) // 2
    y = (img_size - h) // 2 -bbox[1]
    draw.text((x, y), word, font=font, fill=0)
    return img

def find_fixed_font_size(word):
    min_size, max_size = 10, img_size
    best_size = min_size
    target_text_height = round(img_size * max_text_height)

    for size in range(min_size, max_size):
        font = get_font(size)
        bbox = ImageDraw.Draw(Image.new("L", (img_size, img_size))).textbbox((0, 0), word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if h >= target_text_height or w >= img_size * 0.95:
            best_size = size - 1
            break
        best_size = size 
    return get_font(best_size)

def generate_synthetic_data(images_dir):
    words = load_words()
    ensure_dir(images_dir)

    print(f"Saving synthetic images in: {images_dir}")
    for idx, word in enumerate(words):
        font = find_fixed_font_size(word)
        img = render_word(word, font)
        filename = f"synthetic_page00_{idx:03d}_{word}.jpg"
        img.save(images_dir / filename)
    print(f"Skapade {len(words)} syntetiska ordbilder i {images_dir}")
