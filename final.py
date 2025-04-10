import cv2
import numpy as np
import threading
import signal
import time
from flask import Flask, Response

# Flask setup for real-time video streaming
app = Flask(__name__)

# Video capture from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Performance tracking
frame_count = 0
start_time = time.time()
true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0

def make_coordinates(image, line_parameters):
    try:
        if line_parameters is None:
            return None
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)
        if slope == 0 or np.isinf(slope):
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)
    except Exception as e:
        print(f"Error in make_coordinates: {e}")
        return None

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                if abs(x1 - x2) < 5:
                    continue
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope, intercept = parameters
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
            except Exception as e:
                print(f"Error in polyfit: {e}")

    left_fit_avg = np.mean(left_fit, axis=0) if left_fit else None
    right_fit_avg = np.mean(right_fit, axis=0) if right_fit else None
    return left_fit_avg, right_fit_avg

def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[(0, height), (width, height), (width // 2, int(height * 0.6))]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def calculate_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return accuracy, precision, recall, f1_score

def detect_lanes(image):
    global true_positives, false_positives, false_negatives, true_negatives
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cropped_edges = region_of_interest(edges)
        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
        line_image = np.zeros_like(image)
        detected = False

        if lines is not None:
            left_lane, right_lane = average_slope_intercept(image, lines)
            if left_lane is not None:
                left_line = make_coordinates(image, left_lane)
                if left_line is not None:
                    cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)
                    detected = True
            if right_lane is not None:
                right_line = make_coordinates(image, right_lane)
                if right_line is not None:
                    cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)
                    detected = True

        # Update metrics
        if detected:
            true_positives += 1
        else:
            false_negatives += 1

        accuracy, precision, recall, f1_score = calculate_metrics(true_positives, false_positives, false_negatives, true_negatives)
        metrics_text = (f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | "
                        f"Recall: {recall:.2f} | F1 Score: {f1_score:.2f}")

        cv2.putText(image, metrics_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return cv2.addWeighted(image, 0.8, line_image, 1, 0)

    except Exception as e:
        print(f"Error during lane detection: {e}")
        return image

def process_frame(frame):
    global frame_count, start_time
    try:
        processed_frame = detect_lanes(frame)
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        throughput = fps

        text = f"FPS: {fps:.2f} | Throughput: {throughput:.2f} FPS"
        cv2.putText(processed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"{text} | True Positives: {true_positives} | False Negatives: {false_negatives}")
        return processed_frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Real-Time Lane Detection</title></head>
    <body>
        <h1>Real-Time Lane Detection Video Feed</h1>
        <img src="/video_feed" width="720" height="480">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            if processed_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

def shutdown_server():
    print("Shutting down server...")
    cap.release()
    flask_thread.join(timeout=2)
    print("Server shut down successfully.")

def signal_handler(signal, frame):
    print("\nReceived interrupt signal. Cleaning up...")
    shutdown_server()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()

if __name__ == '__main__':
    print("Starting real-time lane detection and streaming...")
    while True:
        pass
