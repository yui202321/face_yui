import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import QTimer, Qt, QSize

# --- 設定と定数 ---
OVERLAY_IMAGE_PATH = "" 

class FaceReplacerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("顔認識＆画像挿入アプリケーション (PyQt6)")
        self.setGeometry(100, 100, 800, 600)
        
        # 状態変数
        self.video_capture = None
        self.is_running = False
        self.overlay_img_cv = None # OpenCV形式の重ね合わせる画像
        
        # MediaPipeの初期化
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        # UIのセットアップ
        self.setup_ui()
        
        # QTimerを設定し、定期的にフレームを更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        """GUI要素を作成し配置します。"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 映像表示エリア (QLabel)
        self.video_label = QLabel("カメラ停止中")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label)

        # 2. コントロールエリア
        control_layout = QHBoxLayout()
        
        # 画像選択セクション
        self.img_path_label = QLabel("重ねる画像: 未選択")
        self.img_path_label.setMinimumWidth(250)
        
        self.select_img_button = QPushButton("画像選択")
        self.select_img_button.clicked.connect(self.select_overlay_image)

        # カメラコントロールセクション
        self.start_button = QPushButton("カメラを起動")
        self.start_button.clicked.connect(self.start_camera)
        
        self.stop_button = QPushButton("カメラを停止")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        control_layout.addWidget(self.img_path_label)
        control_layout.addWidget(self.select_img_button)
        control_layout.addStretch() # 間にスペースを空ける
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(control_layout)

    def select_overlay_image(self):
        """重ね合わせる画像ファイルを選択し、OpenCV形式でロードします。"""
        global OVERLAY_IMAGE_PATH
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "画像ファイルを選択", "", 
            "画像ファイル (*.png *.jpg *.jpeg);;すべてのファイル (*.*)"
        )
        
        if filepath:
            OVERLAY_IMAGE_PATH = filepath
            self.img_path_label.setText(f"重ねる画像: {filepath}")
            
            try:
                # cv2.IMREAD_UNCHANGED でアルファチャンネルを保持
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is None:
                     raise FileNotFoundError("画像の読み込みに失敗しました。")
                self.overlay_img_cv = img
                QMessageBox.information(self, "成功", "画像を正常にロードしました。")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"画像ファイルのロード中にエラーが発生しました: {e}")
                self.overlay_img_cv = None
                self.img_path_label.setText("重ねる画像: ロード失敗")

    def start_camera(self):
        """ウェブカメラを起動し、映像の取得を開始します。"""
        if not self.is_running:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                QMessageBox.critical(self, "エラー", "カメラを開くことができませんでした。")
                return

            self.is_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            # 30 FPSでタイマーを開始
            self.timer.start(33) 
            print("カメラ起動中...")
        
    def stop_camera(self):
        """カメラを停止します。"""
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            
            # QLabelをクリアし、メッセージを表示
            self.video_label.clear()
            self.video_label.setText("カメラ停止中")

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            print("カメラ停止")

    def overlay_transparent_image(self, background, overlay, x, y, w, h):
        """
        背景画像に透明度を考慮して重ね合わせる画像を挿入します。
        (PyQt版でもOpenCVの処理は共通)
        """
        if overlay is None or overlay.size == 0:
            return background
            
        overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

        y1, y2 = max(0, y), min(background.shape[0], y + h)
        x1, x2 = max(0, x), min(background.shape[1], x + w)

        overlay_cropped = overlay_resized[0:y2-y1, 0:x2-x1]
        
        if overlay_cropped.shape[0] <= 0 or overlay_cropped.shape[1] <= 0:
            return background

        if overlay_cropped.shape[2] == 4:
            alpha = overlay_cropped[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            overlay_bgr = overlay_cropped[:, :, :3]
            roi = background[y1:y2, x1:x2]
            
            for c in range(0, 3):
                if roi.shape[:2] == alpha.shape[:2] and roi.shape[:2] == overlay_bgr.shape[:2]:
                    roi[:, :, c] = (roi[:, :, c] * alpha_inv) + \
                                   (overlay_bgr[:, :, c] * alpha)
        else:
            background[y1:y2, x1:x2] = overlay_resized[0:y2-y1, 0:x2-x1]

        return background

    def process_frame(self, frame):
        """
        フレーム内の顔をMediaPipeで検出し、指定された画像を挿入します。
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                # MediaPipeの正規化された座標をピクセル座標に変換
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                x, y = max(0, x), max(0, y)
                
                if self.overlay_img_cv is not None:
                    # 画像を顔の領域に重ね合わせる
                    frame = self.overlay_transparent_image(frame, self.overlay_img_cv, x, y, w, h)
                
        return frame

    def update_frame(self):
        """
        QTimerによって定期的に呼び出され、フレームを更新します。
        """
        if self.is_running and self.video_capture:
            ret, frame = self.video_capture.read()
            
            if ret:
                # カメラの映像を左右反転 (鏡像表示)
                frame = cv2.flip(frame, 1)

                # フレームの処理 (顔認識と画像挿入)
                processed_frame = self.process_frame(frame)
                
                # BGR画像をRGBに変換（PyQt表示のため）
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # QImageに変換
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # QLabelのサイズに合わせてリサイズして表示
                pixmap = QPixmap.fromImage(qt_image)
                # QLabelの幅と高さを取得
                label_w = self.video_label.width()
                label_h = self.video_label.height()
                
                # スケーリングしてQLabelにセット
                self.video_label.setPixmap(pixmap.scaled(
                    label_w, label_h, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                 self.stop_camera()

    def closeEvent(self, event):
        """ウィンドウが閉じられたときに呼ばれる処理"""
        self.stop_camera()
        event.accept()

# --- アプリケーションの実行 ---
if __name__ == "__main__":
    import sys
    
    # PyQt6アプリケーションを作成
    app = QApplication(sys.argv)
    window = FaceReplacerApp()
    window.show()
    
    # アプリケーションの終了を処理
    sys.exit(app.exec())