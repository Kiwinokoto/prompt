from PyPDF2 import PdfReader, PdfWriter

def truncate_pdf(input_path, output_path, num_pages=10):
    # Create PDF reader and writer objects
    reader = PdfReader(input_path)
    writer = PdfWriter()

    # Get total pages in original PDF
    total_pages = len(reader.pages)

    # Determine how many pages to keep
    pages_to_keep = min(num_pages, total_pages)

    # Add pages to new PDF
    for page_num in range(pages_to_keep):
        writer.add_page(reader.pages[page_num])

    # Save the truncated PDF
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

    print(f"Created truncated PDF with {pages_to_keep} pages")

# Usage example
input_file = "./pwc-luxembourg-annual-review-2024.pdf"
output_file = "truncated_document.pdf"
truncate_pdf(input_file, output_file, num_pages=10)