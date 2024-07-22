import os
import cv2
import numpy as np
from skimage.feature import hog
from scipy.spatial.distance import euclidean, cosine


class FaceDetection:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size, flags=1)
        return faces


class FaceAlignment:
    def __init__(self):
        self.reference_points = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        self.output_size = (96, 112)

    def align_face(self, image, landmarks):
        M, _ = cv2.estimateAffinePartial2D(landmarks, self.reference_points)
        aligned_face = cv2.warpAffine(image, M, self.output_size)
        return aligned_face


class FeatureExtraction:
    def __init__(self) -> None:
        pass

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        return features


class FaceComparison:
    def __init__(self) -> None:
        pass

    def compare(self, features1, features2):
        euclidean_distance = euclidean(features1, features2)
        cosine_distance = cosine(features1, features2)
        return euclidean_distance, cosine_distance


def load_stored_images_from_folder(folder_path):
    detector = FaceDetection()
    aligner = FaceAlignment()
    extractor = FeatureExtraction()
    comparator = FaceComparison()
    stored_features = []
    names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            name, _ = os.path.splitext(filename)  # Get the name without extension
            image = cv2.imread(image_path)
            faces = detector.detect_faces(image)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                landmarks = np.array([[x, y], [x + w, y], [x + w / 2, y + h / 2], [x, y + h], [x + w, y + h]], dtype=np.float32)
                aligned_face = aligner.align_face(image, landmarks)
                features = extractor.extract_features(aligned_face)
                stored_features.append(features)
                names.append(name)
    return stored_features, names


def compare_with_live_video(stored_images_with_names):
    detector = FaceDetection()
    aligner = FaceAlignment()
    extractor = FeatureExtraction()
    comparator = FaceComparison()

    stored_features, names = load_stored_images_from_folder('./images')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        print(f"Faces detected: {len(faces)}")
        # Debug: print number of faces detected
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            landmarks = np.array([[x, y], [x + w, y], [x + w / 2, y + h / 2], [x, y + h], [x + w, y + h]], dtype=np.float32)
            aligned_face = aligner.align_face(frame, landmarks)
            features = extractor.extract_features(aligned_face)

            for stored, name in zip(stored_features, names):
                euclidean_distance, cosine_distance = comparator.compare(features, stored)
                print(f"Name: {name}, Euclidean distance: {euclidean_distance}, Cosine distance: {cosine_distance}")
                if euclidean_distance > 0.6:  # Threshold to be adjusted
                    print(f"Euclidean distance: {euclidean_distance}, Cosine distance: {cosine_distance}")  # Debug
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f'Euclidean: {euclidean_distance:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f'Cosine: {cosine_distance:.2f}', (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Live Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


compare_with_live_video('./images')
