# Plan: QR-kod integration för template-identifiering

## 1. Bibliotek och Dependencies

### QR-kod generering
- **`qrcode`** - Senaste versionen för att generera QR-koder
- **`pillow`** (PIL) - För bildhantering (redan används i projektet)

### QR-kod läsning
- **`opencv-python`** (cv2) - Redan används i projektet, har inbyggd QR-kod detektor
- **`pyzbar`** - Alternativ/backup för QR-läsning om opencv inte hittar koden

### PDF manipulation (för att lägga till QR i befintliga PDFs)
- **`PyPDF2`** eller **`pypdf`** - För att läsa befintliga PDFs
- **`reportlab`** - För att skapa nya PDF-sidor med QR-kod
- **`pdf2image`** - Om vi behöver konvertera PDF till bild för att läsa QR-kod

---

## 2. Filstruktur och metadata

### Template metadata utökas med:
```json
{
  "template_id": "swedish_handwriting_tf_20251106_150212",
  "template_type": "text_field",
  "creation_timestamp": "20251106_150212",
  "qr_code_generated": true,
  "qr_code_path": "qr_codes/swedish_handwriting_tf_20251106_150212.png",
  "sentences_source": "swedish_sentences_v2.json",
  "version": "2.0"
}
```

### Ny katalogstruktur:
```
docs/data_collection/generated_templates/
  ├── swedish_handwriting_tf_20251106_150212.json (befintlig metadata)
  ├── swedish_handwriting_tf_20251106_150212.pdf (befintlig PDF)
  ├── qr_codes/
  │   ├── swedish_handwriting_tf_20251106_150212.png (QR-kod som bild)
  │   └── swedish_handwriting_sl_20251107_*.png
  └── updated_pdfs/  (optional - för PDFs med inbäddad QR-kod)
      └── swedish_handwriting_tf_20251106_150212_with_qr.pdf
```

---

## 3. Implementation - Steg för steg

### Steg 1: Skapa QR-kod generator utility
**Fil:** `scripts/utils/qr_code_handler.py`

**Funktionalitet:**
- Generera QR-kod från template_id (timestamp från filnamn)
- QR-kodens data: `{"template_id": "swedish_handwriting_tf_20251106_150212", "type": "text_field"}`
- Spara som PNG i `qr_codes/` katalog
- Returnera path till QR-kod bilden
- Support för olika QR-kod storlekar (small för PDF, large för läsbarhet)

**Input:** Template metadata fil path
**Output:** QR-kod bildfil path

---

### Steg 2: QR-kod läsare utility
**Fil:** `scripts/utils/qr_code_handler.py` (samma fil)

**Funktionalitet:**
- Läs QR-kod från inskannad bild (använd opencv först, fallback till pyzbar)
- Parse QR-kodens data (JSON)
- Returnera template_id
- Hantera flera QR-koder i samma bild (välj rätt baserat på position eller storlek)
- Error handling om QR-kod inte hittas eller är oläsbar

**Input:** Inskannad bildfil path
**Output:** Template metadata dictionary eller None

---

### Steg 3: Retroaktiv QR-kod generering för befintliga mallar
**Fil:** `scripts/data_collection/add_qr_to_existing_templates.py`

**Funktionalitet:**
- Scanna `docs/data_collection/generated_templates` för alla `.json` filer
- För varje template:
  - Läs metadata
  - Extrahera timestamp från filnamn
  - Generera QR-kod med template_id
  - Spara QR-kod som PNG
  - Uppdatera metadata JSON med QR-kod info (lägg till `qr_code_generated: true`, `qr_code_path`)
- Logga vilka templates som processats

**Input:** Template directory
**Output:** QR-kod PNG filer + uppdaterade metadata JSON filer

---

### Steg 4: Lägg till QR-kod i PDF (optional, för framtida användning)
**Fil:** `scripts/data_collection/embed_qr_in_pdf.py`

**Funktionalitet:**
- Läs befintlig PDF
- Skapa ny sida eller overlay med QR-kod i hörnet (t.ex. övre högra hörnet)
- Spara som ny PDF i `updated_pdfs/` eller skriv över original
- Position: Konsistent plats där den inte stör text (t.ex. 10mm från övre högra hörnet)
- Storlek: Lagom stor för att vara läsbar men inte påträngande (t.ex. 2x2 cm)

**Input:** PDF path + QR-kod PNG path
**Output:** Uppdaterad PDF med inbäddad QR-kod

---

### Steg 5: Modifiera template_generator
**Filer:** 
- `scripts/data_collection/template_generator_tf.py`
- `scripts/data_collection/template_generator_sl.py`

**Ändringar:**
- Lägg till `--template-id` argument (optional, default: auto-generate från timestamp)
- Efter PDF-generering:
  - Generera QR-kod automatiskt
  - Lägg till QR-kod i PDF innan sparning
  - Uppdatera metadata med QR-info
  - Spara QR-kod PNG separat
- Metadata ska innehålla:
  - `template_id`
  - `template_type` (text_field eller single_line)
  - `qr_code_path`
  - `sentences_source` (om applicable)

---

### Steg 6: Modifiera orchestrator för QR-kod läsning
**Fil:** `scripts/data_processing/orchestrator/main.py`

**Ändringar:**
- Lägg till `--template-id` argument (optional)
- Lägg till `--auto-detect-template` flag (default: True)
- Vid processing av originals:
  1. Om `--template-id` angiven: Använd den
  2. Annars om `--auto-detect-template`:
     - Läs QR-kod från första bilden i katalogen
     - Hitta matchande template metadata
     - Använd den template-typen för processing
  3. Fallback: Använd senaste template (current behavior)

**Ny funktion:**
```python
detect_template_from_image(image_path) -> template_metadata
  - Läs QR-kod från bild
  - Parse template_id
  - Hitta matchande .json metadata fil
  - Returnera metadata
```

---

### Steg 7: Uppdatera `config/paths.py`
**Fil:** `config/paths.py`

**Ändringar:**
- Lägg till `QR_CODES` path under `DocsPaths`
- Uppdatera `get_single_line_metadata()` och `get_text_field_metadata()`:
  - Ta emot optional `template_id` parameter
  - Om angiven: Hitta specifik template baserat på ID
  - Annars: Returnera senaste (current behavior)

**Ny funktion:**
```python
get_template_by_id(template_id: str, template_type: str) -> Path
  - Sök efter template med specifikt ID
  - Returnera metadata path
  - Raise error om inte hittas
```

---

## 4. QR-kod Design och innehåll

### QR-kodens data format:
```json
{
  "template_id": "swedish_handwriting_tf_20251106_150212",
  "type": "text_field",
  "version": "2.0"
}
```

### QR-kod specifikationer:
- **Format:** JSON string (kompakt, lätt att parse)
- **Error correction level:** H (High) - 30% redundans för att klara skanning av delvis skadade QR-koder
- **Storlek i PDF:** 2x2 cm (tillräckligt stor för läsbarhet)
- **Position i PDF:** Övre högra hörnet, 10mm från kant
- **PNG storlek:** 300x300 pixels för god kvalitet

---

## 5. Backwards compatibility och migration

### För befintliga templates (utan QR):
1. Kör retroaktiv QR-generering script (`add_qr_to_existing_templates.py`)
2. Genererar QR-koder för alla befintliga templates
3. Uppdaterar metadata filer
4. Orchestrator kan fortfarande använda `--template-id` manuellt för dessa

### För framtida templates:
- QR-kod genereras automatiskt vid template creation
- Inbäddad direkt i PDF
- Orchestrator kan auto-detect från inskannade bilder

### Fallback strategy:
- Om QR-kod inte kan läsas: Använd `--template-id` argument
- Om varken QR eller argument: Använd senaste template (current behavior)
- Logga alltid vilken template som används

---

## 6. Testing och validering

### Test scenarios:
1. **Generera QR för befintliga templates** - Verifiera att alla får QR-koder
2. **Läs QR från ren bild** - Testa QR-läsning på genererad template PDF
3. **Läs QR från inskannad bild** - Testa med verkliga skanningar (olika kvalitet)
4. **Auto-detect i orchestrator** - Verifiera att rätt template hittas
5. **Manuell template-id override** - Testa att `--template-id` fungerar
6. **Fallback till senaste** - Testa att system fungerar om QR saknas

### Validering:
- Alla templates har unika template_id
- QR-koder är läsbara från PDF och skanningar
- Metadata uppdateras korrekt
- Orchestrator hittar rätt template konsekvent

---

## 7. Implementation ordning

1. **Fas 1:** QR utilities (`qr_code_handler.py`)
2. **Fas 2:** Retroaktiv QR-generering för befintliga templates
3. **Fas 3:** Uppdatera template generators för framtida templates
4. **Fas 4:** Uppdatera orchestrator för QR-detection
5. **Fas 5:** Testing och validering
6. **Fas 6:** (Optional) Embed QR in PDF för befintliga templates

---

## 8. Potential issues och lösningar

### Problem: QR-kod oläsbar från skanning
**Lösning:** 
- Öka QR-kod storlek
- Använd högre error correction
- Fallback till manuell `--template-id`

### Problem: Flera QR-koder i samma bild
**Lösning:**
- Läs alla QR-koder
- Välj den som matchar template format
- Logga varning om flera hittas

### Problem: Template ID collision
**Lösning:**
- Timestamp har sekund-precision, unlikely collision
- Om collision: Lägg till random suffix

---

## Nuvarande implementation (Enkel lösning)

**Implementerat 2025-11-07:**
- Footer med filnamn i PDF templates (text_field och single_line)
- `--template-id` argument i orchestrator
- Uppdaterade `get_single_line_metadata()` och `get_text_field_metadata()` i `config/paths.py`
- Manuell specifikation av template via kommandoradsargument

**Framtida QR-kod implementation kommer att bygga på denna grund.**
