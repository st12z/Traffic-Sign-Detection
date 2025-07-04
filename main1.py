import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model và danh sách các lớp
def load_traffic_classifier():
    model = load_model('model1.h5')
    classes = {
        1: 'Tối đa 20 km/h',
        2: 'Tối đa 30 km/h',
        3: 'Speed limit (50km/h)',
        4: 'Speed limit (60km/h)',
        5: 'Speed limit (70km/h)',
        6: 'Speed limit (80km/h)',
        7: 'End of speed limit (80km/h)',
        8: 'Speed limit (100km/h)',
        9: 'Speed limit (120km/h)',
        10: 'No passing',
        11: 'No passing veh over 3.5 tons',
        12: 'Right-of-way at intersection',
        13: 'Priority road',
        14: 'Yield',
        15: 'Stop',
        16: 'No vehicles',
        17: 'Veh > 3.5 tons prohibited',
        18: 'No entry',
        19: 'General caution',
        20: 'Dangerous curve left',
        21: 'Dangerous curve right',
        22: 'Double curve',
        23: 'Bumpy road',
        24: 'Slippery road',
        25: 'Road narrows on the right',
        26: 'Road work',
        27: 'Traffic signals',
        28: 'Pedestrians',
        29: 'Children crossing',
        30: 'Bicycles crossing',
        31: 'Beware of ice/snow',
        32: 'Wild animals crossing',
        33: 'End speed + passing limits',
        34: 'Turn right ahead',
        35: 'Turn left ahead',
        36: 'Ahead only',
        37: 'Go straight or right',
        38: 'Go straight or left',
        39: 'Keep right',
        40: 'Keep left',
        41: 'Roundabout mandatory',
        42: 'End of no passing'
    }
    return model, classes

# Giao diện GUI
def init_gui():
    global top, label, sign_image

    top = tk.Tk()
    top.geometry('800x600')
    top.title('Traffic Sign Classification')
    top.configure(background='#CDCDCD')

    label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
    sign_image = Label(top)
    heading = Label(top, text="Predict Traffic Sign", pady=20, font=('arial', 20, 'bold'))
    heading.configure(background='#CDCDCD', foreground='#364156')

    upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
    upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

    upload.pack(side=tk.BOTTOM, pady=50)
    sign_image.pack(side=tk.BOTTOM, expand=True)
    label.pack(side=tk.BOTTOM, expand=True)
    heading.pack()

    exit_app = Button(top, text="Exit App", command=top.destroy, padx=10, pady=5)
    exit_app.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    exit_app.pack(side=tk.BOTTOM)

    return top

# Hàm phân loại ảnh
def classify(file_path, model, classes, label):
    image = Image.open(file_path)
    image = image.resize((30, 30))  # Resize về đúng kích thước mô hình cần

    image = np.array(image)

    if image.shape[-1] != 3:  # Nếu không phải ảnh RGB
        image = np.stack((image,) * 3, axis=-1)

    image = image.astype('float32') / 255.0  # Chuẩn hóa pixel
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch

    pred_probs = model.predict(image)
    pred = np.argmax(pred_probs, axis=1)

    sign = classes.get(pred[0] + 1, "Unknown")
    print(sign)
    label.configure(foreground='#011638', text=sign)

# Hiển thị nút phân loại
def show_classify_button(file_path, model, classes):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path, model, classes, label), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

# Hàm upload ảnh
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')

        show_classify_button(file_path, model, classes)
    except Exception as e:
        print("Error:", e)

# Chạy ứng dụng
if __name__ == "__main__":
    model, classes = load_traffic_classifier()
    top = init_gui()
    top.mainloop()
