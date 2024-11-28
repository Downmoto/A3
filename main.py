import cv2 as cv
from ultralytics import YOLO
import os
import easyocr
import re
import csv

# Initialize YOLO and EasyOCR
model = YOLO("./GroupSevenTrained.pt")
reader = easyocr.Reader(['en'])

# Directory containing images
image_sets = "./datasets/images"
output_csv = "./datasets/results.csv"

# CSV headers
headers = ["Image Name", "Detected Text", "Confidence", "Preprocessing Method", "Bounding Box", "Score"]

# Open CSV for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

    # Iterate through all images in the directory
    for image_name in os.listdir(image_sets):
        image_path = os.path.join(image_sets, image_name)
        image = cv.imread(image_path)

        if image is None:
            print(f"Failed to open {image_name} at {image_path}")
            continue

        # YOLO inference
        results = model(image)[0]

        # Process each detected plate
        for plate in results.boxes.data.tolist():
            x1, y1, x2, y2, score, id = plate

            cropped = image[int(y1):int(y2), int(x1):int(x2)]

            if cropped.size == 0:
                continue

            # Preprocessing Steps
            preprocessing_steps = []

            # Step 1: Original Grayscale
            cropped_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            preprocessing_steps.append(("Grayscale", cropped_gray))

            # Step 2: Adaptive Threshold
            cropped_threshold = cv.adaptiveThreshold(
                cropped_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
            )
            preprocessing_steps.append(("Adaptive Threshold", cropped_threshold))

            # Step 3: Gaussian Blur + Threshold
            blurred = cv.GaussianBlur(cropped_threshold, (3, 3), 0)
            preprocessing_steps.append(("Blurred + Threshold", blurred))

            # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(cropped_gray)
            preprocessing_steps.append(("CLAHE", enhanced))

            # Step 5: Resize each preprocessing step
            for i, (desc, step) in enumerate(preprocessing_steps):
                resized_step = cv.resize(step, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                preprocessing_steps[i] = (desc, resized_step)

            # Use OCR after each preprocessing step and store the best result
            best_text = ""
            best_confidence = 0.0
            best_preprocessing = ""

            for desc, processed_image in preprocessing_steps:
                text_results = reader.readtext(
                    processed_image, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
                )

                for (_, text, prob) in text_results:
                    if prob > best_confidence:
                        best_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        best_confidence = prob
                        best_preprocessing = desc

            if best_text:
                # Write results to CSV
                bounding_box = f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                writer.writerow([image_name, best_text, f"{best_confidence:.2f}", best_preprocessing, bounding_box, f"{score:.2f}"])

                # Annotate original image
                text_size = cv.getTextSize(best_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = int(x1)
                text_y = int(y1) - 10
                background_tl = (text_x, text_y - text_size[1])
                background_br = (text_x + text_size[0], text_y + 5)

                cv.rectangle(image, background_tl, background_br, (0, 0, 0), -1)
                cv.putText(image, best_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save annotated image (optional)
        output_image_path = f"./output/{image_name}"
        os.makedirs("./output", exist_ok=True)
        cv.imwrite(output_image_path, image)

print(f"Results saved to {output_csv}")
