import cv2
import numpy as np

def detect_camera_tampering(input_video, output_video):
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"❌ Error opening video file: {input_video}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    prev_frame = None
    total_area = frame_width * frame_height
    frame_index = 0  # Track frame number

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tampering_detected = False

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Morphological operations to reduce noise
            motion_mask = cv2.erode(motion_mask, np.ones((4, 4), np.uint8), iterations=1)
            motion_mask = cv2.dilate(motion_mask, np.ones((9, 9), np.uint8), iterations=3)

            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                print(f"Frame {frame_index} - Contour Area: {area}")
                if area > 0.4 * total_area:
                    tampering_detected = True
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if tampering_detected:
            print(f"⚠️ Tampering detected at frame {frame_index}")
            text = "Tampering Detected"
            org = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_color = (255, 255, 255)  # White
            bg_color = (0, 0, 255)        # Red background for visibility

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            top_left = (org[0] - 10, org[1] - text_height - 10)
            bottom_right = (org[0] + text_width + 10, org[1] + 10)

            # Draw rectangle background
            cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)

            # Put text on top
            cv2.putText(frame, text, org, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        prev_frame = gray.copy()
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"\n✅ Tampering detection complete. Output saved to: {output_video}")

# Usage
if __name__ == "__main__":
    detect_camera_tampering("cropped_video.mp4", "tampering_output.mp4")
