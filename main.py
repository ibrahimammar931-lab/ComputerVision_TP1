from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Import the UI class generated from your design.py
# Make sure design.py is in the same folder as main.py
# ─────────────────────────────────────────────────────────────────────────────
from design import Ui_MainWindow   # adjust if your class name differs


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        # Internal state
        self.img_bgr   = None   # original image loaded by OpenCV (BGR)
        self.img_path  = "C:\\Users\\ibrah\\PycharmProjects\\FirstProject\\licensed-image.webp"

        # ── Connect buttons ────────────────────────────────────────────────
        # Browse / load image  →  (you can add a browse button; here we also
        # auto-load the hard-coded path on startup for convenience)
        # If your UI has a Browse button, wire it like:
        # self.pushButton_1.clicked.connect(self.get_image)

        # Channel display buttons
        self.pushButton_2.clicked.connect(self.showRedChannel)
        self.pushButton_3.clicked.connect(self.showGreenChannel)
        self.pushButton_4.clicked.connect(self.showBlueChannel)

        # Colour histogram
        self.pushButton_5.clicked.connect(self.show_HistColor)

        # Gray image + gray histogram
        # Wire "Valider" button to show_UpdatedImgGray
        # and "Afficher Histogramme gris" to show_HistGray
        # Update the button names below if they differ in your design.py
        try:
            self.pushButton_6.clicked.connect(self.show_UpdatedImgGray)
        except AttributeError:
            pass   # button not present in this design

        try:
            self.pushButton_7.clicked.connect(self.show_HistGray)
        except AttributeError:
            pass

        # Auto-load the default image at startup
        self._load_image(self.img_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ─────────────────────────────────────────────────────────────────────────

    def convert_cv_qt(self, cv_image):
        """Convert an OpenCV image (BGR or Gray) to QPixmap."""
        if len(cv_image.shape) == 2:
            # Grayscale → convert to 3-channel for QImage
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        h, w, ch = cv_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            cv_image.data, w, h, bytes_per_line,
            QtGui.QImage.Format_BGR888
        )
        return QPixmap.fromImage(qt_image)

    def _set_pixmap(self, label, pixmap):
        """Scale a pixmap to fit inside a QLabel while keeping aspect ratio."""
        label.setPixmap(
            pixmap.scaled(
                label.width(), label.height(),
                aspectRatioMode=1   # Qt.KeepAspectRatio
            )
        )

    def _make_figure_in_label(self, fig, label):
        """Render a matplotlib figure into a QLabel."""
        fig.tight_layout()
        fig.savefig("_tmp_fig.png", dpi=100)
        plt.close(fig)
        pixmap = QPixmap("_tmp_fig.png")
        self._set_pixmap(label, pixmap)

    def showDimensions(self):
        """Display image dimensions in textEdit_5."""
        if self.img_bgr is None:
            return
        if len(self.img_bgr.shape) == 3:
            h, w, ch = self.img_bgr.shape
        else:
            h, w = self.img_bgr.shape
            ch = 1
        self.textEdit_5.setText(
            f"Hauteur : {h} px\nLargeur : {w} px\nCanaux  : {ch}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Image loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_image(self, file_path):
        """Internal helper – loads an image and refreshes the UI."""
        img = cv2.imread(file_path)
        if img is None:
            QtWidgets.QMessageBox.warning(
                self, "Erreur", f"Impossible de charger :\n{file_path}"
            )
            return
        self.img_bgr = img
        # Show original image in label_12
        pixmap = self.convert_cv_qt(self.img_bgr)
        self._set_pixmap(self.label_12, pixmap)
        # Show dimensions
        self.showDimensions()

    def get_image(self):
        """Open a file dialog, load the selected image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
        )
        if file_path:
            self.img_path = file_path
            self._load_image(file_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Channel extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _make_channel_image(self, channel_index):
        """
        Return a BGR image where only the chosen channel is non-zero.
        channel_index: 0=Blue, 1=Green, 2=Red  (OpenCV BGR order)
        """
        if self.img_bgr is None:
            return None
        zeros = np.zeros_like(self.img_bgr[:, :, channel_index])
        channels = [zeros, zeros, zeros]
        channels[channel_index] = self.img_bgr[:, :, channel_index]
        return cv2.merge(channels)

    def showRedChannel(self):
        """Extract red channel and display in label_13."""
        img_ch = self._make_channel_image(2)   # index 2 = Red in BGR
        if img_ch is not None:
            self._set_pixmap(self.label_13, self.convert_cv_qt(img_ch))

    def showGreenChannel(self):
        """Extract green channel and display in label_14."""
        img_ch = self._make_channel_image(1)   # index 1 = Green
        if img_ch is not None:
            self._set_pixmap(self.label_14, self.convert_cv_qt(img_ch))

    def showBlueChannel(self):
        """Extract blue channel and display in label_15."""
        img_ch = self._make_channel_image(0)   # index 0 = Blue
        if img_ch is not None:
            self._set_pixmap(self.label_15, self.convert_cv_qt(img_ch))

    # ─────────────────────────────────────────────────────────────────────────
    # Colour histogram
    # ─────────────────────────────────────────────────────────────────────────

    def show_HistColor(self):
        """Compute colour histogram, save as PNG, display in label_16."""
        if self.img_bgr is None:
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ('b', 'g', 'r')
        labels = ('Blue', 'Green', 'Red')
        for i, (col, lbl) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([self.img_bgr], [i], None, [256], [0, 256])
            ax.plot(hist, color=col, label=lbl)
        ax.set_title("Histogramme Couleur")
        ax.set_xlabel("Intensité")
        ax.set_ylabel("Fréquence")
        ax.legend()
        fig.savefig("Color_Histogram.png", dpi=100)
        plt.close(fig)
        self._set_pixmap(self.label_16, QPixmap("Color_Histogram.png"))

    # ─────────────────────────────────────────────────────────────────────────
    # Contrast & Brightness readers
    # ─────────────────────────────────────────────────────────────────────────

    def getContrast(self):
        """Read contrast value from textEdit_7 (alpha). Default = 1.0."""
        try:
            val = float(self.textEdit_7.toPlainText().strip())
        except (ValueError, AttributeError):
            val = 1.0
        return val

    def getBrightness(self):
        """Read brightness value from textEdit_8 (beta). Default = 0."""
        try:
            val = float(self.textEdit_8.toPlainText().strip())
        except (ValueError, AttributeError):
            val = 0.0
        return val

    # ─────────────────────────────────────────────────────────────────────────
    # Gray image
    # ─────────────────────────────────────────────────────────────────────────

    def _get_gray_updated(self):
        """Convert to gray then apply contrast/brightness adjustments."""
        if self.img_bgr is None:
            return None
        alpha = self.getContrast()
        beta  = self.getBrightness()
        img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        img_updated = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
        return img_updated

    def show_UpdatedImgGray(self):
        """Show adjusted grayscale image in label_17."""
        img_gray = self._get_gray_updated()
        if img_gray is not None:
            self._set_pixmap(self.label_17, self.convert_cv_qt(img_gray))

    # ─────────────────────────────────────────────────────────────────────────
    # Gray histogram
    # ─────────────────────────────────────────────────────────────────────────

    def calc_HistGray(self):
        """Compute grayscale histogram array for the adjusted image."""
        img_gray = self._get_gray_updated()
        if img_gray is None:
            return None
        return cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    def show_HistGray(self):
        """Save gray histogram as PNG and display in label_18."""
        hist = self.calc_HistGray()
        if hist is None:
            return
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(hist, color='gray')
        ax.set_title("Histogramme Niveaux de Gris")
        ax.set_xlabel("Intensité")
        ax.set_ylabel("Fréquence")
        fig.savefig("Gray_Histogram.png", dpi=100)
        plt.close(fig)
        self._set_pixmap(self.label_18, QPixmap("Gray_Histogram.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())