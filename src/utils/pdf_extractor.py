import os
import pdfplumber
import json
import csv
from tqdm import tqdm
import config

PDF_DIRECTORY = config.PDF_DIRECTORY
JSON_OUTPUT_FILE = config.INPUT_JSON_PATH
CSV_OUTPUT_FILE = config.CSV_OUTPUT_FILE

def extract_text_from_pdf(pdf_path):
    """
    Verilen bir PDF dosyasının yolunu alır ve içindeki tüm metni birleştirerek döndürür.
    
    Args:
        pdf_path (str): PDF dosyasının tam yolu.
        
    Returns:
        tuple: (çıkarılan metin, sayfa sayısı) veya hata durumunda (None, 0).
    """
    full_text = ""
    page_count = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                # Sayfadan metin çıkar, eğer sayfa boşsa veya sadece resim içeriyorsa None dönebilir.
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n" # Sayfalar arasına boşluk ekle
        return full_text.strip(), page_count
    except Exception as e:
        print(f"Hata: '{os.path.basename(pdf_path)}' dosyası işlenemedi. Hata mesajı: {e}")
        return None, 0

def process_pdf_directory(input_dir, output_json_path, output_csv_path):
    """
    Belirtilen klasördeki tüm PDF dosyalarını işler ve sonuçları JSON ve CSV olarak kaydeder.
    
    Args:
        input_dir (str): PDF dosyalarının bulunduğu klasör.
        output_json_path (str): Çıktı JSON dosyasının yolu.
        output_csv_path (str): Çıktı CSV dosyasının yolu.
    """
    if not os.path.isdir(input_dir):
        print(f"Hata: '{input_dir}' klasörü bulunamadı.")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"'{input_dir}' klasöründe hiç PDF dosyası bulunamadı.")
        return

    all_data = []
    
    # tqdm ile ilerleme çubuğu ekliyoruz
    for filename in tqdm(pdf_files, desc="PDF'ler İşleniyor"):
        pdf_path = os.path.join(input_dir, filename)
        content, pages = extract_text_from_pdf(pdf_path)
        
        if content:
            data_item = {
                "filename": filename,
                "page_count": pages,
                "content": content
            }
            all_data.append(data_item)

    # Verileri JSON olarak kaydet
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"Veriler başarıyla '{output_json_path}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"JSON dosyasına yazılırken hata oluştu: {e}")

    # Verileri CSV olarak kaydet
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            if all_data:
                # CSV başlıklarını ilk veri öğesinin anahtarlarından al
                writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                writer.writeheader()
                writer.writerows(all_data)
        print(f"Veriler başarıyla '{output_csv_path}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"CSV dosyasına yazılırken hata oluştu: {e}")


if __name__ == "__main__":
    process_pdf_directory(PDF_DIRECTORY, JSON_OUTPUT_FILE, CSV_OUTPUT_FILE)