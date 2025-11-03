import io
import requests
from PIL import Image
import flet as ft

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint

def main(page: ft.Page):
    page.title = "Food Classification (ResNet50 • MindSpore)"
    page.window_width = 930
    page.window_height = 620
    page.horizontal_alignment = "center"
    page.vertical_alignment = "start"
    page.padding = 20

    picked_img = ft.Image(width=320, height=320, fit=ft.ImageFit.COVER, border_radius=12)
    result_text = ft.Text("", size=16, weight=ft.FontWeight.W_600)
    conf_text = ft.Text("", size=14)
    status = ft.Text("")
    pb = ft.ProgressBar(width=320, visible=False)

    def on_pick_result(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        f = e.files[0]
        # show preview
        with open(f.path, "rb") as fh:
            img_bytes = fh.read()
        picked_img.src_base64 = io.BytesIO(img_bytes).getvalue().hex()
        picked_img.update()
        # call API
        pb.visible = True; status.value = "Predicting…"; page.update()
        try:
            files = {"file": (f.name, open(f.path, "rb"), "image/jpeg")}
            r = requests.post(API_URL, files=files, timeout=30)
            r.raise_for_status()
            data = r.json()
            result_text.value = f"Prediction: {data['prediction']}"
            conf_text.value = f"Confidence: {data['confidence']} %"
            status.value = "Done."
        except Exception as ex:
            result_text.value = ""
            conf_text.value = ""
            status.value = f"Error: {ex}"
        finally:
            pb.visible = False
            page.update()

    fp = ft.FilePicker(on_result=on_pick_result)
    page.overlay.append(fp)

    pick_btn = ft.ElevatedButton(
        "Select Image",
        icon=ft.icons.IMAGE_OUTLINED,
        on_click=lambda _: fp.pick_files(
            allow_multiple=False,
            allowed_extensions=["jpg", "jpeg", "png"],
        ),
    )

    card = ft.Card(
        content=ft.Container(
            padding=20,
            content=ft.Column(
                [
                    ft.Row([pick_btn]),
                    picked_img,
                    pb,
                    result_text,
                    conf_text,
                    status,
                ],
                tight=False,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ),
    )

    page.add(card)

if __name__ == "__main__":
    ft.app(target=main)
