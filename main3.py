import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model và danh sách các lớp theo dataset3
def load_traffic_classifier():
    model = load_model('model3.h5')
    classes = {
        0: 'Speed limit (5km/h)',
        1: 'Speed limit (15km/h)',
        2: 'Speed limit (30km/h)',
        3: 'Speed limit (40km/h)',
        4: 'Speed limit (50km/h)',
        5: 'Speed limit (60km/h)',
        6: 'Speed limit (70km/h)',
        7: 'speed limit (80km/h)',
        8: 'Dont Go straight or left',
        9: 'Dont Go straight or Right',
        10: 'Dont Go straight',
        11: 'Dont Go Left',
        12: 'Dont Go Left or Right',
        13: 'Dont Go Right',
        14: 'Dont overtake from Left',
        15: 'No Uturn',
        16: 'No Car',
        17: 'No horn',
        18: 'Speed limit (40km/h)',
        19: 'Speed limit (50km/h)',
        20: 'Go straight or right',
        21: 'Go straight',
        22: 'Go Left',
        23: 'Go Left or right',
        24: 'Go Right',
        25: 'keep Left',
        26: 'keep Right',
        27: 'Roundabout mandatory',
        28: 'watch out for cars',
        29: 'Horn',
        30: 'Bicycles crossing',
        31: 'Uturn',
        32: 'Road Divider',
        33: 'Traffic signals',
        34: 'Danger Ahead',
        35: 'Zebra Crossing',
        36: 'Bicycles crossing',
        37: 'Children crossing',
        38: 'Dangerous curve to the left',
        39: 'Dangerous curve to the right',
        40: 'Unknown1',
        41: 'Unknown2',
        42: 'Unknown3',
        43: 'Go right or straight',
        44: 'Go left or straight',
        45: 'Unknown4',
        46: 'ZigZag Curve',
        47: 'Train Crossing',
        48: 'Under Construction',
        49: 'Unknown5',
        50: 'Fences',
        51: 'Heavy Vehicle Accidents',
        52: 'Unknown6',
        53: 'Give Way',
        54: 'No stopping',
        55: 'No entry',
        56: 'Unknown7',
        57: 'Unknown8',
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
    image = image.resize((30, 30))  # Resize đúng kích thước model

    image = np.array(image)

    # Nếu ảnh không có 3 kênh màu, chuyển thành 3 kênh
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)

    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension

    pred_probs = model.predict(image)
    pred = np.argmax(pred_probs, axis=1)

    sign = classes.get(pred[0], "Unknown")
    print(f"Predicted: {sign}")
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
