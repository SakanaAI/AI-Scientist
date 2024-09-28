import os
import json
from dotenv import load_dotenv
import argparse
import openai
from ai_scientist.perform_review import load_paper, perform_review
from PyPDF2 import PdfReader, PdfWriter

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-4o-2024-05-13", help="Model name")
parser.add_argument("paper", help="Path to the PDF file")
parser.add_argument("-n", type=int, default=1, help="Number of reviews")
parser.add_argument("--num_reflections", type=int, default=5, help="Number of reflections")
parser.add_argument("--num_fs_examples", type=int, default=1, help="Number of FS examples")
parser.add_argument("--num_reviews_ensemble", type=int, default=5, help="Number of reviews in ensemble")
parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")
parser.add_argument("--max-pages", type=int, default=0, help="Maximum number of pages of the paper to process. Useful to exclude appendixes. Will truncate any pages after the number you specify.")
parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
args = parser.parse_args()

openai.api_key = args.openai_api_key

# Truncate the PDF file if necessary
if args.max_pages > 0:
    print(f"Truncating {args.paper} to {args.max_pages} pages")
    input_pdf = PdfReader(open(args.paper, "rb"))
    output_pdf = PdfWriter()

    for page_num in range(min(args.max_pages, len(input_pdf.pages))):
        output_pdf.add_page(input_pdf.pages[page_num])

    with open("temp.pdf", "wb") as f:
        output_pdf.write(f)
else:
    # Copy the file to temp.pdf
    with open(args.paper, "rb") as src_file, open("temp.pdf", "wb") as dest_file:
        dest_file.write(src_file.read())

# Repeat the perform_review function args.n times
for i in range(args.n):
    print(f"Starting Review {i+1} of {args.n}")
    # Get the review dict of the review
    review = perform_review(
        load_paper("temp.pdf"),
        args.model,
        openai.OpenAI(),
        num_reflections=args.num_reflections,
        num_fs_examples=args.num_fs_examples,
        num_reviews_ensemble=args.num_reviews_ensemble,
        temperature=args.temperature,
    )

    # Output review as JSON
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir + "/" + os.path.basename(args.paper).replace(".pdf", f"_{i+1}.json")
    with open(output_file, "w") as f:
        json.dump(review, f)

# Cleanup temp.pdf
print("Cleaning up temporary files")
os.remove("temp.pdf")
print("Done")