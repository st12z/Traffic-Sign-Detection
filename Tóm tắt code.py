# Hàm tải và xử lý ảnh
def load_images_from_directory(path, image_size=(30, 30)):
    images = os.listdir(path)
    data = []
    labels = []
    class_id = int(os.path.basename(path))  # Lấy ID lớp từ tên thư mục

    for image_filename in images:
        try:
            image = Image.open(os.path.join(path, image_filename))
            image = image.resize(image_size) # thay đổi kích thước ảnh về kích thước 30x30 pixel
            image = np.array(image) # Chuyển ảnh thành 1 mảng numpy (30,30,3) 3 :số kênh màu. Mỗi pixel có 3 giá trị R,G,B nằm trong khoảng 0 đến 255
            image = image / 255.0  # Mỗi pixel của ảnh RGB có giá trị từ 0 đến 255, chia cho 255 để đưa các giá trị về khoảng [0,1]
            data.append(image)
            labels.append(class_id)
        except:
            print(f"Lỗi khi tải ảnh: {os.path.join(path, image_filename)}")

    return data, labels

# Tải toàn bộ ảnh và nhãn
data = [] # danh sách chứa các ảnh xử lý
labels = [] # danh sách chứa nhãn tương ứng của mỗi ảnh
num_classes = 43 # 43 lớp 
current_path = os.getcwd()

for class_id in range(num_classes):
    path = os.path.join(current_path, 'data/Train', str(class_id))
    class_data, class_labels = load_images_from_directory(path)
    data.extend(class_data)
    labels.extend(class_labels)

data = np.array(data) # chuyển tất cả các ảnh vè numpy
labels = np.array(labels) # chuyển tất cả các ảnh về numpy

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# truyền vào 4 tham số: 
    # data: chứa tất cả các ảnh đầu vào
    #labels: chứa tất cả các nhãn
    # chia dữ liệu: 20% dùng để kiểm tra, 80% dùng để huấn luyện
    # X_train: 80% ảnh dùng để huấn luyện
    # X_test: 20% dùng để kiểm tra
    # y_train: 80% nhãn tương ứng với X_train
    # y_test: 20% nhãn tương ứng với X_test


# One-hot encoding
# Chuyển nhãn từ số nguyên thành dạng vector one-hot dạng nhị phân
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

# Tạo ImageDataGenerator với tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train_one_hot, batch_size=32)
test_generator = test_datagen.flow(X_test, y_test_one_hot)

# Xây dựng mô hình CNN với BatchNormalization
#Sử dụng mô hình tuần tự mỗi lớp, nằm sau layer trước đó
model = Sequential()

#Layer đầu tiên: 
# 64 bộ lọc kích thước 3x3, kích hoạt activation "relu"
#  padding='same' : giữ nguyên kích thước ảnh đầu ra
# BatchNormalization(): chuẩn hóa dữ liệu sau mỗi batch giúp tăng tốc độ học và ổn định
# Dropout(0.1): giảm overfitting bằng cách tắt 10% neuron 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(0.1))

#Layer thức hai:
# Thêm 1 lớp Conv tương tự
# MaxPool: giảm kích thước không gia ảnh(height,width): chọn max trên vùng 2x2
# Dropout(0.25): tăng cường giảm overfitting.
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

# Chuyển đổi feature maps
# GlobalAveragePooling2D(): tính trung bình giá trị toàn bộ mỗi features map 
model.add(GlobalAveragePooling2D())
# Dense(256, activation='relu'): Tạo một lớp ẩn fully-connected với 256 neuron 
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Lớp đầu ra của mô hình
# Gồm 43 lớp
# activation="softmax": Chuyển đầu ra thành xác suất cho mỗi lớp. Tổng bằng 1
# Ví dụ : [0.01,0.02,0.95,...] : ảnh thuộc lớp số 2 với độ tin cậy 95%
model.add(Dense(num_classes, activation='softmax'))