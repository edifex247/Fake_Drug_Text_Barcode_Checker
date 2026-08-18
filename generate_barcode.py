import barcode
from barcode.writer import ImageWriter

# Demonstration barcode from our reference database
barcode_number = "615110000001"

# Generate Code 128 barcode
code = barcode.get(
    "code128",
    barcode_number,
    writer=ImageWriter()
)

# Save the barcode image
filename = code.save("demo_paracetamol_barcode")

print(f"Barcode image created successfully: {filename}.png")
