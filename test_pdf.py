import fitz

pdf_path = "AM_FM_Modulation_Recognition_Paper.pdf"
doc = fitz.open(pdf_path)
print(f"PDF总页数: {len(doc)}")

for page_num in range(min(5, len(doc))):
    page = doc[page_num]
    text = page.get_text()
    print(f"\n{'='*50}")
    print(f"第 {page_num + 1} 页:")
    print('='*50)
    print(text[:1000])

doc.close()
