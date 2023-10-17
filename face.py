import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

class FaceRecognizer:

    def __init__(self):
        self.model = None
        self.class_names = []

    def normalize_images(self, root_folder):
        skipped_folders = []

        for subdir, _, files in os.walk(root_folder):
            unrecognized_format = False
            for file in files:
                filepath = os.path.join(subdir, file)
                filename, ext = os.path.splitext(filepath)
                output_filepath = filename + ".jpg"

                if ext.lower() in [".png", ".bmp", ".tiff", ".gif", ".jpg", ".jpeg", ".JPG"]:
                    with Image.open(filepath) as img:
                        img.convert("RGB").save(output_filepath, "JPEG", quality=90)
                        if filepath != output_filepath:
                            os.remove(filepath)
                else:
                    unrecognized_format = True

            if unrecognized_format:
                skipped_folders.append(subdir)

        return skipped_folders

    def create_face_model(self, data_folder):
        X = []
        y = []
        label = 0

        subfolders = sorted([f.path for f in os.scandir(data_folder) if f.is_dir()])

        for subdir in subfolders:
            self.class_names.append(os.path.basename(subdir))
            for file in os.listdir(subdir):
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, (100, 100))
                    X.append(image)
                    y.append(label)
            label += 1

        X = np.array(X) / 255.0
        X = X.reshape(-1, 100, 100, 1)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(label, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=4, validation_data=(X_test, y_test))

    def predict_class(self, image_path):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(gray_image, (100, 100))
        normalized_image = resized_image / 255.0
        reshaped_image = normalized_image.reshape(-1, 100, 100, 1)
        predictions = self.model.predict(reshaped_image)[0]
        max_probability = np.max(predictions)

        if max_probability > 0.8:
            return np.argmax(predictions), max_probability
        else:
            return None, max_probability

    def capture_and_predict(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("No se pudo abrir la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame.")
                break

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (100, 100))
            normalized_image = resized_image / 255.0
            reshaped_image = normalized_image.reshape(-1, 100, 100, 1)
            predictions = self.model.predict(reshaped_image)[0]
            max_probability = np.max(predictions)

            if max_probability > 0.8:
                predicted_class_index = np.argmax(predictions)
                label = f"Clase: {self.class_names[predicted_class_index]}, Probabilidad: {max_probability * 100:.2f}%"
            else:
                label = "No se pudo hacer una predicción confiable."

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root_folder = "Fotos_todos"
    
    recognizer = FaceRecognizer()
    skipped = recognizer.normalize_images(root_folder)
    print("Carpetas con formatos no reconocidos:", skipped)

    recognizer.create_face_model(root_folder)
    recognizer.capture_and_predict()
