import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from config.paths import get_latest_version, get_version_images, ensure_dir

WORDS_PATH = Path(__file__).parent / "syntetic_words.json"
IMG_SIZE = 384
FONT_PATH = "config/fonts/ReenieBeanie-Regular.ttf"  

def load_words():
    with open(WORDS_PATH, encoding="utf-8") as f:
        return json.load(f)

def get_font(font_size):
    return ImageFont.truetype(FONT_PATH, font_size)

def render_word(word, font):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (IMG_SIZE - w) // 2
    y = (IMG_SIZE - h) // 2
    draw.text((x, y), word, font=font, fill=0)
    return img

def find_max_font_size(word):
    min_size, max_size = 10, IMG_SIZE
    while min_size < max_size:
        size = (min_size + max_size + 1) // 2
        font = get_font(size)
        bbox = ImageDraw.Draw(Image.new("L", (IMG_SIZE, IMG_SIZE))).textbbox((0, 0), word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= IMG_SIZE and h <= IMG_SIZE:
            min_size = size
        else:
            max_size = size - 1
    return get_font(min_size)

def main():
    words = load_words()
    version_dir = get_latest_version()  # t.ex. Path('.../trocr_ready_data/v3')
    images_dir = get_version_images(version_dir.name)  # t.ex. Path('.../trocr_ready_data/v3/images')
    ensure_dir(images_dir)

    for idx, word in enumerate(words):
        font = find_max_font_size(word)
        img = render_word(word, font)
        filename = f"synthetic_page00_{idx:03d}_{word}.jpg"
        img.save(images_dir / filename)
    print(f"Skapade {len(words)} syntetiska ordbilder i {images_dir}")

if __name__ == "__main__":
    main()