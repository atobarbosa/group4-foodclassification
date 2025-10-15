import flet as ft
import requests
import os

# Backend server URL from Sprint#3 part 2.py
SERVER_URL = "http://127.0.0.1:8000/predict"

def main(page: ft.Page):
    """
    This function builds and runs the Flet user interface.
    """
    page.title = "Food Classification System"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 400
    page.window_height = 550
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # --- UI State ---
    # Store the path of the selected image file
    page.client_storage.set("selected_image_path", None)
    
    # --- UI Components ---
    title = ft.Text("🍕 Food Classifier", size=30, weight=ft.FontWeight.BOLD)
    
    # Text to show prediction results or status messages
    prediction_text = ft.Text(
        "Upload an image to classify!", 
        size=16, 
        italic=True, 
        text_align=ft.TextAlign.CENTER
    )
    
    # Image display area
    food_image = ft.Image(
        src="https://via.placeholder.com/224x224.png?text=Your+Image+Here",
        width=224,
        height=224,
        border_radius=ft.border_radius.all(10),
        fit=ft.ImageFit.CONTAIN,
    )
    
    # --- Functions / Event Handlers ---
    def on_file_selected(e: ft.FilePickerResultEvent):
        """
        Callback function that is triggered when a user selects a file.
        """
        if e.files:
            selected_file = e.files[0]
            # Store the path for later use
            page.client_storage.set("selected_image_path", selected_file.path)
            # Update the on-screen image to show the selected picture
            food_image.src = selected_file.path
            prediction_text.value = f"Selected: {os.path.basename(selected_file.path)}"
            page.update()
    
    def classify_image(e):
        """
        Sends the selected image to the FastAPI server for prediction.
        """
        image_path = page.client_storage.get("selected_image_path")
        
        if not image_path:
            prediction_text.value = "Please select an image first."
            page.update()
            return
            
        prediction_text.value = "🔎 Classifying, please wait..."
        page.update()
        
        try:
            # Open the image file in binary-read mode and send it
            with open(image_path, "rb") as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(SERVER_URL, files=files)
                response.raise_for_status()
                
                # Display the prediction from the server's response
                data = response.json()
                prediction = data.get('prediction', 'Unknown')
                confidence = data.get('confidence', 0)
                prediction_text.value = f"✅ Prediction: {prediction}\nConfidence: {confidence:.2%}"
                
        except requests.exceptions.ConnectionError:
            prediction_text.value = "❌ Error: Cannot connect to server. Is it running?"
        except requests.exceptions.RequestException as req_err:
            prediction_text.value = f"❌ Request error: {str(req_err)}"
        except Exception as ex:
            prediction_text.value = f"❌ Error: {str(ex)}"
        
        page.update()
    
    # File picker for selecting images
    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)
    
    # Button to open file picker
    upload_button = ft.ElevatedButton(
        "📁 Select Image",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: file_picker.pick_files(
            allowed_extensions=["jpg", "jpeg", "png"],
            dialog_title="Select a food image"
        )
    )
    
    # Button to classify the image
    classify_button = ft.ElevatedButton(
        "🔍 Classify",
        icon=ft.icons.SEARCH,
        on_click=classify_image
    )
    
    # --- Layout ---
    page.add(
        ft.Container(
            content=ft.Column(
                [
                    title,
                    ft.Divider(height=20, color="transparent"),
                    food_image,
                    ft.Divider(height=10, color="transparent"),
                    prediction_text,
                    ft.Divider(height=20, color="transparent"),
                    upload_button,
                    ft.Divider(height=10, color="transparent"),
                    classify_button,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10
            ),
            padding=20,
        )
    )

# Run the app
if __name__ == "__main__":
    ft.app(target=main)