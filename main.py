from ultralytics import YOLO
import os

# Load the best-trained model
model = YOLO("C:/pinokio/api/automatic1111.git/app/runs/detect/train3/weights/best.pt")

# Function to perform inference and display the result
def test_model_on_images(image_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all images in the input folder
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    
    for image_name in images:
        # Load the image
        image_path = os.path.join(image_folder, image_name)
        print(f"Processing {image_name}...")

        # Run inference
        results = model.predict(image_path)

        # results is a list of result objects, loop through them
        for result in results:
            # Convert the result to a Pandas DataFrame for inspection
            df = result.to_df()  # Convert to DataFrame
            print(df)  # Display the result dataframe with class, box, and confidence

            # Show the image with detected bounding boxes
            result.show()  # This will display the image with bounding boxes

            # Save the resulting image with detections (optional)
            save_path = os.path.join(output_folder, f"detected_{image_name}")
            result.save(save_path)
            print(f"Result saved at {save_path}")

    print("Inference completed on all images.")

# Set your image folder path and output folder path
image_folder = "images"  # Replace with the folder containing test images
output_folder = "test_outputs"      # Replace with the folder where you want to save results

# Run the inference and visualize results
test_model_on_images(image_folder, output_folder)
