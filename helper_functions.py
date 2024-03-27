import numpy as np
import cv2
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QGraphicsScene, QVBoxLayout, QPushButton, QSlider, 
                             QSizePolicy, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLineEdit, QLabel, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QMutex
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QPushButton, QLineEdit, QLabel, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QProgressBar
"import QLINEF"
from PyQt5.QtCore import QLineF
#import ghrapics line item
from PyQt5.QtWidgets import QGraphicsLineItem
#qpen
from PyQt5.QtGui import QPen

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QSlider, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QIcon
import cv2
import time
from calibration import CalibrationDialog
import math
from ROI import ROIDialog

def process_frame(cv_img, display_width, display_height):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w

    # Convert to QImage using a copy of the data
    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    # Scale only if the dimensions are different
    if w != display_width or h != display_height:
        q_img = q_img.scaled(display_width, display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return q_img

def color_map(velocity, max_velocity):
    normalized_velocity = min(velocity / max_velocity, 1)
    blue_component = int(normalized_velocity * 255)
    red_component = int((1 - normalized_velocity) * 255)
    return red_component, 0, blue_component

def draw_flow(img, flow, winsize=30, real_time_fps=30, show_fps=False, frame_count=0, start_time=None, prev_avg_velocity=0, alpha=0.1, conv_factor=1.0, color_map_on=False):
    """
    Draws the optical flow on the given image.

    Parameters:
    img (numpy.ndarray): The input image (grayscale or color) on which the flow vectors will be drawn.
    flow (numpy.ndarray): The calculated optical flow vectors for each point in the image.
    winsize (int, optional): The size of the window for calculating optical flow. Default is 30.
    real_time_fps (float, optional): The frames per second of the video input for scaling the flow vectors. Default is 30.
    show_fps (bool, optional): If True, the current FPS will be displayed on the image. Default is False.
    frame_count (int, optional): The current count of frames processed, used for FPS calculation. Default is 0.
    start_time (float, optional): The start time in seconds for FPS calculation. If None, FPS will not be calculated. Default is None.
    prev_avg_velocity (float, optional): The previous average velocity, used for calculating the exponential moving average. Default is 0.
    alpha (float, optional): The alpha value for the exponential moving average calculation. Default is 0.1.
    conv_factor (float, optional): A conversion factor for adjusting the velocity scale. Default is 1.0.
    color_map_on (bool, optional): If True, a color map will be applied to the flow lines based on velocity. Default is False.

    Returns:
    tuple: A tuple containing:
        - vis (numpy.ndarray): The image with flow vectors drawn.
        - avg_velocity (float): The current average velocity calculated from the flow vectors.
    
    Description:
    The function calculates the flow vectors' magnitudes and directions from the 'flow' parameter.
    It then draws these vectors on the 'img' as lines, indicating the motion detected in the image.
    If 'show_fps' is True, it also calculates and displays the current frames per second on the image.
    The function uses an exponential moving average to smooth out the velocity values over time.
    """  
    
    # Get height (h) and width (w) of the image for grid calculation
    h, w = img.shape[:2]

    # Calculate the step size for sampling flow vectors; it depends on the window size (winsize)
    step = max(2, winsize // 2)

    # Generate a grid of (x, y) coordinates for sampling the flow.
    # The grid points are spaced 'step' units apart and cover the whole image.
    # The grid is then flattened into 1D arrays for 'x' and 'y'.
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(np.int16)

    # Sample the optical flow vectors (fx, fy) at each grid point.
    # 'fx' and 'fy' represent motion in the x and y directions, respectively.
    fx, fy = flow[y, x].T

    # Calculate the magnitude of the flow vector (velocity) at each sampled point.
    # This represents the speed and scale of movement.
    velocities = np.hypot(fx, fy) * real_time_fps * conv_factor

    # Compute the average velocity across all sampled points for a general sense of motion.
    current_velocity = np.mean(velocities) if velocities.size > 0 else 0

    # Calculate the Exponential Moving Average of the velocity to smooth fluctuations over time.
    avg_velocity = (current_velocity * alpha) + (prev_avg_velocity * (1 - alpha))

    # Create an array of line segments to represent the flow. Each line starts at (x, y) and
    # ends at (x + fx, y + fy), indicating the direction and magnitude of flow.
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int16(lines + 0.5)

    # Convert the image to a BGR color image if it's not already, to draw colored lines.
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw each line on the image. Each line represents the optical flow vector at one grid point.
    for ((x1, y1), (x2, y2)) in lines:
        color = (0, 255, 0)  # Set the line color to green
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)

    # If FPS (Frames Per Second) display is enabled, compute and display it on the image.
    if show_fps and start_time is not None:
        current_fps = frame_count / (time.time() - start_time)  # Calculate current FPS
        cv2.putText(vis, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv2.putText(vis, f"Avg Velocity: {avg_velocity:.2f}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

    # Return the final image with optical flow vectors drawn and the current average velocity.
    return vis, avg_velocity

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_slider_signal = pyqtSignal(int)
    mutex = QMutex()

    def __init__(self, display_width, display_height, video_path, process_frame):
        super().__init__()
        self.display_width = display_width
        self.display_height = display_height
        self.video_path = video_path
        self.process_frame = process_frame  # The function to process each frame
        self.running = False
        self.paused = False
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        print("Video opened: Frame count:", self.frame_count)

    def run(self):
        while self.cap.isOpened():
            if self.running:
                ret, cv_img = self.read_frame()
                if ret:
                    # Use the provided process_frame function
                    p = self.process_frame(cv_img, self.display_width, self.display_height)
                    self.change_pixmap_signal.emit(p)
                    if not self.paused:
                        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        slider_value = int((current_frame / self.frame_count) * 100)
                        self.update_slider_signal.emit(slider_value)
                else:
                    print("No more frames to read.")
                    break
            self.msleep(10)

    def read_frame(self):
        self.mutex.lock()
        try:
            ret, frame = self.cap.read()
        finally:
            self.mutex.unlock()
        return ret, frame

    def stop_video(self):
        self.running = False
        self.paused = False
        self.mutex.lock()
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, cv_img = self.cap.read()
            if ret:
                p = self.process_frame(cv_img, self.display_width, self.display_height)
                self.change_pixmap_signal.emit(p)
            self.update_slider_signal.emit(0)
        finally:
            self.mutex.unlock()

    def pause(self):
        self.running = False
        self.paused = True
        print("Video paused")

    def start_video(self):
        self.running = True
        self.paused = False
        print("Video started")
        if not self.isRunning():
            self.start()

    def seek(self, value):
        self.mutex.lock()
        try:
            frame_no = int((value / 100) * self.frame_count)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, cv_img = self.cap.read()
            if ret:
                p = self.process_frame(cv_img, self.display_width, self.display_height)
                self.change_pixmap_signal.emit(p)
        finally:
            self.mutex.unlock()
        if self.running:
            self.paused = False

class VideoPlayer(QWidget):
    def __init__(self, parent=None, video_path=None, process_frame=None):
        super().__init__(parent)
        self.setWindowTitle("Video Player")
        self.display_width = None
        self.display_height = None
        self.init_ui()
        self.thread = None
        if video_path:
            self.load_video(video_path)

    def init_ui(self):
        # UI Setup
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)

        # Button layout
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.start_button = QPushButton()
        self.start_button.setIcon(QIcon("media/play.png"))
        self.button_layout.addWidget(self.start_button)

        self.pause_button = QPushButton()
        self.pause_button.setIcon(QIcon("media/pause.png"))
        self.button_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon("media/stop.png"))
        self.button_layout.addWidget(self.stop_button)

        self.slider = QSlider(Qt.Horizontal)
        self.button_layout.addWidget(self.slider)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)

    def load_video(self, video_path):
        if self.thread is not None:
            # Stop the current thread safely
            self.thread.stop_video()
            self.thread.terminate()  # Terminate the current thread
            self.thread.wait()  # Wait for the thread to fully terminate

        # Release the current video capture object
        if self.thread and self.thread.cap:
            self.thread.cap.release()

        # Open the video file temporarily to get its dimensions
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from video file.")
            cap.release()
            return

        self.display_height, self.display_width, _ = frame.shape
        print("Video dimensions:", self.display_width, self.display_height)
        cap.release()

        # Initialize a new thread for the new video
        self.thread = VideoThread(self.display_width, self.display_height, video_path, process_frame)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_slider_signal.connect(self.update_slider)
        self.start_button.clicked.connect(self.thread.start_video)
        self.pause_button.clicked.connect(self.thread.pause)
        self.stop_button.clicked.connect(self.thread.stop_video)
        self.slider.valueChanged.connect(self.slider_changed)

        # set fisrt frame to the video label
        ret, cv_img = self.thread.read_frame()
        if ret:
            p = process_frame(cv_img, self.display_width, self.display_height)
            self.update_image(p)

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    @pyqtSlot(int)
    def update_slider(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)

    def slider_changed(self, value):
        if self.thread:
            self.thread.seek(value)
    def get_current_frame(self):
        if self.thread and self.thread.cap.isOpened():
            ret, frame = self.thread.cap.read()
            if ret:
                return frame
        return None

class ButtonLineEdit(QWidget):
    """A widget that combines a line edit and a button."""
    def __init__(self, icon_path=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(0)  # Set spacing to zero between widgets

        self.input = QLineEdit()
        self.input.setStyleSheet("QLineEdit { border: 1px solid gray; border-radius: 2px; }")
        layout.addWidget(self.input, 1)  # The line edit will expand

        self.set_button = QPushButton("...")
        if icon_path:
            self.set_button.setIcon(QIcon(icon_path))
            self.set_button.setText("")  # Remove text if icon is set
        self.set_button.setStyleSheet("QPushButton { border: 1px solid gray; border-radius: 2px; }")
        self.set_button.setFixedSize(self.set_button.sizeHint())  # Set a fixed size for the button

        layout.addWidget(self.set_button)
        layout.setContentsMargins(0, 0, 0, 0)  # Set contents margins to zero
        self.setLayout(layout)


class OptionsMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)


        # Video
        video_item = QTreeWidgetItem(self.tree)
        video_item.setText(0, "Video")

        # Video Path
        video_path_item = QTreeWidgetItem(video_item)
        video_path_item.setText(0, "Path")
        self.video_path_edit = ButtonLineEdit()
        self.tree.setItemWidget(video_path_item, 1, self.video_path_edit)
        self.video_path_edit.set_button.clicked.connect(self.open_video_file_dialog)



        fps_item = QTreeWidgetItem(video_item)
        fps_item.setText(0, "FPS")
        self.fps_edit = ButtonLineEdit('media/gear.png')
        self.tree.setItemWidget(fps_item, 1, self.fps_edit)

        # Preprocessing
        preprocessing_item = QTreeWidgetItem(self.tree)
        preprocessing_item.setText(0, "Preprocessing")
        contrast_enhancement_item = QTreeWidgetItem(preprocessing_item)
        contrast_enhancement_item.setText(0, "Contrast")
        self.contrast_enhancement_checkbox = QCheckBox()
        #set checkbox checked True
        self.contrast_enhancement_checkbox.setChecked(True)
        self.tree.setItemWidget(contrast_enhancement_item, 1, self.contrast_enhancement_checkbox)

        # ROI
        roi_item = QTreeWidgetItem(self.tree)
        roi_item.setText(0, "ROI")
        coordinates_item = QTreeWidgetItem(roi_item)
        coordinates_item.setText(0, "Coordinates")
        self.roi_set_widget = ButtonLineEdit('media/gear.png')
        self.tree.setItemWidget(coordinates_item, 1, self.roi_set_widget)
        self.roi_set_widget.set_button.clicked.connect(self.open_roi_dialog)

        # Calibration
        calibration_item = QTreeWidgetItem(self.tree)
        calibration_item.setText(0, "Calibration")

        #add reference for calibration in pxls
        reference_item = QTreeWidgetItem(calibration_item)
        reference_item.setText(0, "Reference pxls")
        self.reference_set_widget = ButtonLineEdit('media/gear.png')
        self.tree.setItemWidget(reference_item, 1, self.reference_set_widget)
        self.reference_set_widget.set_button.clicked.connect(self.open_calibration_dialog)

    
        #add reference for calibration in cm
        reference_cm_item = QTreeWidgetItem(calibration_item)
        reference_cm_item.setText(0, "Reference cm")
        self.reference_cm_set_widget = QLineEdit()
        self.tree.setItemWidget(reference_cm_item, 1, self.reference_cm_set_widget)


        conv_factor_item = QTreeWidgetItem(calibration_item)
        conv_factor_item.setText(0, "Conv. Factor")
        self.calibration_set_widget = QLineEdit()
        self.tree.setItemWidget(conv_factor_item, 1, self.calibration_set_widget)


        # Optical Flow
        optical_flow_item = QTreeWidgetItem(self.tree)
        optical_flow_item.setText(0, "Optical Flow")
        self.add_optical_flow_properties(optical_flow_item)

        self.layout.addWidget(self.tree)


        # Create a horizontal layout for the button and the progress bar
        self.process_layout = QHBoxLayout()

        # Process Video Button
        self.process_video_button = QPushButton("Process Video")
        self.process_video_button.clicked.connect(self.process_video)
        self.process_layout.addWidget(self.process_video_button)

        # Loading Bar (Progress Bar)
        self.loading_bar = QProgressBar()
        self.loading_bar.setMaximum(100)  # Assuming 100 as the maximum value
        self.loading_bar.setMinimum(0)  # Assuming 0 as the minimum value
        self.loading_bar.setTextVisible(False)  # Set to True if you want to show percentage
        self.process_layout.addWidget(self.loading_bar)
        #hide the progress bar
        self.loading_bar.hide()


        # Add the horizontal layout to the main layout
        self.layout.addLayout(self.process_layout)


        line_edit_stylesheet = "QLineEdit { border: 0.5px solid gray; border-radius: 2px; }"
        self.setStyleSheet(line_edit_stylesheet)


        # This will allow the first column to resize freely
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)

    def open_roi_dialog(self):
        current_frame = self.parent.video_player.get_current_frame()
        if current_frame is not None:
            roi_dialog = ROIDialog(self,current_frame)
            roi_dialog.coords.connect(self.update_roi_coords)
            roi_dialog.exec_()
        else:
            print("No current frame available for ROI selection.")

    def open_calibration_dialog(self):
        current_frame = self.parent.video_player.get_current_frame()
        if current_frame is not None:
            calibration_dialog = CalibrationDialog(self, current_frame)
            calibration_dialog.lineLengthSignal.connect(self.update_reference_length)
            calibration_dialog.exec_()
        else:
            print("No current frame available for calibration.")

    def update_roi_coords(self, coords):
        # convert dict coortd to plain string and set it to the line edit
        self.roi_set_widget.input.setText(str(coords))

    def update_reference_length(self, length):
        # Update the ButtonLineEdit with the length
        self.reference_set_widget.input.setText(str(length))

    def add_optical_flow_properties(self, parent_item):
        user_friendly_names = {
            "Pyramid Scale": "0.5",
            "Levels": "3",
            "Window Size": "15",
            "Iterations": "3",
            "Poly N": "5",
            "Poly Sigma": "1.2",
            "Flags": "0"
        }
        for name, default_value in user_friendly_names.items():
            prop_item = QTreeWidgetItem(parent_item)
            prop_item.setText(0, name)
            prop_edit = QLineEdit(default_value)
            self.tree.setItemWidget(prop_item, 1, prop_edit)

    def open_video_file_dialog(self):
        # Open file dialog for video files
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            # Set the selected file path in the line edit
            self.video_path_edit.input.setText(file_path)
            self.parent.video_player.load_video(file_path)

            #set fps_edit to the fps of the video from self.parent.video_player.frame_count
            self.fps_edit.input.setText(str(self.parent.video_player.thread.frame_rate))

    def process_video(self):
        #unhide the progress bar
        self.loading_bar.show()
        path=self.video_path_edit.input.text()
        #convert the fps to float
        fps=float(self.fps_edit.input.text())
        #get the coordinates of the roi, convert string to dict
        roi_dict=eval(self.roi_set_widget.input.text())
        #get the reference conversion factor
        conv_factor=float(self.calibration_set_widget.text())

        #pull the screen size from the parent
        display_width=self.parent.video_player.display_width
        display_height=self.parent.video_player.display_height

        OF_process_video(path, fps, roi_dict,conv_factor)

def OF_process_video(video_path, real_time_fps, roi_dict, conv_factor):
    
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to read video file.")
        return

    
    ret, prev = cap.read()
    frame_height, frame_width = prev.shape[:2]
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the ROI based on scale factor
    w_roi, h_roi = roi_dict['w_roi'], roi_dict['h_roi']
    x_roi, y_roi = roi_dict['x_roi'], roi_dict['y_roi']
    use_roi = True 

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 10, (frame_width, frame_height))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_count, prev_avg_velocity, start_time = 0, 0, time.time()

    # Main video processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] if use_roi else frame
        roi_frame_orig = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Apply contrast enhancement
        frame_YUV = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YUV)
        frame_YUV[:, :, 0] = cv2.equalizeHist(frame_YUV[:, :, 0])
        roi_frame = cv2.cvtColor(frame_YUV, cv2.COLOR_YUV2BGR)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow Processing
        if frame_count == 1: prev_gray = gray
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=100, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
        prev_gray = gray.copy()

        vis, prev_avg_velocity = draw_flow(roi_frame_orig, flow, winsize=100, real_time_fps=real_time_fps, show_fps=True, frame_count=frame_count, start_time=start_time, prev_avg_velocity=prev_avg_velocity, conv_factor=conv_factor, color_map_on=True)
        
        # Insert the processed ROI back into the frame and convert to BGR
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        bgr_frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = vis

        cv2.imshow('Processed Video', bgr_frame)
        out.write(bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
