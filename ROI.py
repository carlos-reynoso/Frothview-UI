import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,
                             QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsItem)
from PyQt5.QtGui import QImage, QPixmap, QBrush, QPen, QPainter
from PyQt5.QtCore import QRectF, QSizeF, Qt

import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QDialog, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPen, QBrush, QColor, QCursor
from PyQt5.QtCore import Qt, QRectF


class DraggableRectItem(QGraphicsRectItem):
    class ResizableHandle(QGraphicsRectItem):
        def __init__(self, parent=None):
            super().__init__(-5, -5, 10, 10, parent)  # Small square handle
            self.setBrush(QBrush(QColor(0, 0, 255, 127)))  # Blue color handle
            self.setCursor(QCursor(Qt.SizeFDiagCursor))
            self.setPen(QPen(Qt.NoPen))



    def __init__(self, rect):
        super().__init__(rect)
        self.setBrush(QBrush(Qt.transparent))  # Hollow square
        self.setPen(QPen(Qt.black, 2))
        self.setFlags(QGraphicsRectItem.ItemIsMovable | QGraphicsRectItem.ItemIsSelectable)
        self.default_brush = QBrush(Qt.transparent)  # Store the default brush color
        self.setBrush(self.default_brush)  # Set the default brush color
        self.handle = self.ResizableHandle(self)
        self.updateHandlePosition()

    def mousePressEvent(self, event):
        # Bring this item to the front
        self.scene().clearSelection()  # Optional: clear other selections
        self.setSelected(True)  # Optional: select this item
        current_max_z = max([item.zValue() for item in self.scene().items() if item != self], default=0)
        self.setZValue(current_max_z + 1)
        
        # Handle resizing if the handle is clicked
        if self.handle.isUnderMouse():
            self.resizing = True
        else:
            self.resizing = False
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            # Calculate the new rectangle based on the mouse position
            newRect = QRectF(self.rect().topLeft(), event.pos()).normalized()
            self.setRect(newRect)
            self.updateHandlePosition()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.resizing = False
        super().mouseReleaseEvent(event)

    def updateHandlePosition(self):
        # Place the handle at the bottom-right corner of the rect item
        handlePos = self.rect().bottomRight()
        self.handle.setPos(handlePos.x() - 5, handlePos.y() - 5)  # Adjust position for the square handle


    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemSelectedChange:
            if value:
                # Set the brush color to transparent blue when selected
                self.setBrush(QBrush(QColor(0, 0, 255, 127)))
            else:
                # Revert the brush color to its default when deselected
                self.setBrush(self.default_brush)
        return super().itemChange(change, value)



from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QDialog, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap,QWheelEvent
from PyQt5.QtCore import QRectF, pyqtSignal

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.025
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

class ROIDialog(QDialog):
    coords = pyqtSignal(dict)  # Define a signal to emit the line length
    def __init__(self, parent=None, image=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Selection")
        self.setFixedSize(800, 600)

        # Initialize QGraphicsScene and QGraphicsView
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)

        # Convert the NumPy array (image) to QImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        q_img = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        # Add a square (DraggableRectItem) to the scene
        self.add_square()

        # Layout for buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch(1)  # This will push the buttons to the right

        # Accept Button
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.accept_button)

        # Apply Button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply)
        self.button_layout.addWidget(self.apply_button)

        # Cancel Button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.view)
        self.main_layout.addLayout(self.button_layout)
        self.setLayout(self.main_layout)

    def apply(self):
        self.emit_coords()

    def accept(self):
        self.emit_coords()
        super().accept()

    def emit_coords(self):
        # Assuming that 'self.view' is your QGraphicsView
        items = self.view.scene().items()
        rect_item = next((item for item in items if isinstance(item, DraggableRectItem)), None)
        pixmap_item = next((item for item in items if isinstance(item, QGraphicsPixmapItem)), None)

        if rect_item and pixmap_item:
            rect = rect_item.rect()

            # Get corners in scene coordinates
            top_left_scene = rect_item.mapToScene(rect.topLeft())
            bottom_right_scene = rect_item.mapToScene(rect.bottomRight())

            # If the image is transformed in the scene, apply the inverse transformation
            transform = pixmap_item.sceneTransform().inverted()[0]
            top_left_image = transform.map(top_left_scene)
            bottom_right_image = transform.map(bottom_right_scene)

            # Normalize the coordinates
            x_roi = max(0, int(top_left_image.x()))
            y_roi = max(0, int(top_left_image.y()))
            x2 = min(pixmap_item.pixmap().width(), int(bottom_right_image.x()))
            y2 = min(pixmap_item.pixmap().height(), int(bottom_right_image.y()))

            # Calculate width and height
            w_roi = x2 - x_roi
            h_roi = y2 - y_roi

            self.coords.emit({'x_roi': x_roi, 'y_roi': y_roi, 'w_roi': w_roi, 'h_roi': h_roi})

            print(f"ROI coordinates and size: x_roi={x_roi}, y_roi={y_roi}, w_roi={w_roi}, h_roi={h_roi}")




    def add_square(self):
        # Assuming DraggableRectItem is defined somewhere else
        # It should be a QGraphicsRectItem with added functionality for dragging and resizing
        rect_item = DraggableRectItem(QRectF(0, 0, 100, 100))
        self.scene.addItem(rect_item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # You need to load an image using OpenCV for this to work
    # Replace 'path_to_your_image.jpg' with the path to your actual image file
    image = cv2.imread('img.jpg')
    if image is not None:
        dialog = ROIDialog(current_frame=image)
        if dialog.exec_():
            print("ROI accepted")
        else:
            print("ROI rejected")
    else:
        print("Failed to load the image.")
    sys.exit(app.exec_())
