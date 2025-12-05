"""
Generate QR codes for GitHub figure links.
"""

import qrcode
from pathlib import Path


def generate_qr_codes():
    """Generate QR codes pointing to GitHub raw URLs."""

    # Placeholders - replace with your actual values
    GITHUB_USERNAME = "REPLACE_GH_USERNAME"
    REPO_NAME = "REPLACE_REPO_NAME"
    BRANCH = "main"

    figures = [
        'predicted_vs_actual.png',
        'poster_ready_figure.png',
        'correlation_heatmap.png'
    ]

    Path('qr_codes').mkdir(exist_ok=True)

    print("\n📱 Generating QR codes...")

    for fig_name in figures:
        # Construct raw GitHub URL
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/figures/{fig_name}"

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(raw_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        qr_path = f"qr_codes/{fig_name.replace('.png', '_qr.png')}"
        img.save(qr_path)

        print(f"  ✓ {qr_path}")
        print(f"    URL: {raw_url}")

        # ASCII QR code preview
        print("\n  ASCII QR Preview:")
        qr.print_ascii(invert=True)
        print()

    print("✓ QR codes generated")


if __name__ == "__main__":
    generate_qr_codes()
