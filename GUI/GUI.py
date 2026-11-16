"""PyQt5 launcher for the book examples."""
import codecs
from pathlib import Path
import sys
from uuid import uuid4

try:
    from PyQt5 import QtCore as C, QtGui as G, QtWidgets as W
except ImportError:
    raise SystemExit("PyQt5 eksik. Sanal ortamda arayüz/requirements-gui.txt dosyasını kurun.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calistir import ROOT, project_scripts
from dogrula import CAMERA_CASES

CHAPTERS = dict(zip("23467", (
    "Benzerlik algoritmaları", "Bulanık mantık", "Makine öğrenmesi",
    "Doğal dil işleme", "Bilgisayarlı görü",
)))
FIELDS = {
    "lev.py": (("İlk kelime", "ahmet"), ("İkinci kelime", "mehmet")),
    "proje.py": (("Yol (0–90)", "60"), ("Hava (0–10)", "8"),
                 ("Hız (30–150)", "70"), ("Deneyim (0–20)", "12")),
    "uyelikDeneme.py": (("Boy (cm)", "170"),),
    "DesicionTrees.py": (("Cinsiyet (kadın/erkek)", "kadın"), ("Yaş", "35"),
                         ("Akyuvar sayısı", "6000"), ("Kronik rahatsızlık (evet/hayır)", "hayır")),
    "RandomForest.py": (("Deneyim (yıl)", "5"),),
    "KNN.py": (("X koordinatı", "12"), ("Y koordinatı", "12")),
    "KNN-test.py": (("X koordinatı", "12"), ("Y koordinatı", "12")),
    "yenidenBoyutlandirma.py": (("Yeni genişlik", "320"),),
    "ConsolePrediction.py": (("Haber metni", "Bugün spor müsabakası düzenlendi."),),
    "cumle.py": (("İlk cümle", "çay kahve fincan"), ("İkinci cümle", "kahve sıcak içecek")),
    "kelime.py": (("İlk kelime", "çay"), ("İkinci kelime", "kahve")),
}


class Launcher(W.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KODLAB • Proje Atölyesi")
        self.resize(1120, 760)
        self.process = C.QProcess(self)
        self.process.setProcessChannelMode(C.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.process.started.connect(self.send_initial_input)
        self.script = None
        self.preview_dir = None
        self.closing = False
        self.stopping = False
        self.fields = []

        self.search = W.QLineEdit(placeholderText="Projelerde ara…")
        self.search.textChanged.connect(self.filter_projects)
        self.tree = W.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAccessibleName("Proje listesi")
        self.tree.currentItemChanged.connect(self.select_project)
        self.tree.itemDoubleClicked.connect(lambda *_: self.run_button.click())
        sidebar = W.QWidget()
        side = W.QVBoxLayout(sidebar)
        side.addWidget(W.QLabel("KİTAP PROJELERİ"))
        side.addWidget(self.search)
        side.addWidget(self.tree)

        self.title = W.QLabel("Bir proje seç")
        self.title.setObjectName("title")
        self.note = W.QLabel()
        self.note.setWordWrap(True)
        self.form = W.QFormLayout()
        self.run_button = W.QPushButton("▶  Çalıştır")
        self.run_button.setObjectName("run")
        self.run_button.clicked.connect(self.run_project)
        self.stop_button = W.QPushButton("Durdur")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.folder = W.QPushButton("Proje klasörü")
        self.folder.clicked.connect(self.open_folder)
        self.browser = W.QPushButton("Yerel sayfayı aç")
        self.browser.clicked.connect(
            lambda: G.QDesktopServices.openUrl(C.QUrl("http://127.0.0.1:5000/"))
        )
        self.browser.hide()
        buttons = W.QHBoxLayout()
        for button in (self.run_button, self.stop_button, self.folder, self.browser):
            buttons.addWidget(button)
        buttons.addStretch()

        self.log = W.QPlainTextEdit(readOnly=True)
        self.log.document().setMaximumBlockCount(2500)
        self.log.setAccessibleName("Çalıştırma çıktısı")
        self.images = W.QComboBox()
        self.images.currentIndexChanged.connect(self.show_preview)
        self.picture = W.QLabel(alignment=C.Qt.AlignCenter)
        self.picture.setText("Görsel çıktı burada gösterilir.")
        self.picture.setMinimumSize(1, 1)
        self.picture.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Ignored)
        preview = W.QWidget()
        panel = W.QVBoxLayout(preview)
        panel.addWidget(self.images)
        panel.addWidget(self.picture, 1)
        self.tabs = W.QTabWidget()
        self.tabs.addTab(self.log, "Çıktı")
        self.tabs.addTab(preview, "Önizleme")
        self.status = W.QLabel("Hazır")

        detail = W.QWidget()
        layout = W.QVBoxLayout(detail)
        for widget in (self.title, self.note):
            layout.addWidget(widget)
        layout.addLayout(self.form)
        layout.addLayout(buttons)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)
        split = W.QSplitter()
        split.addWidget(sidebar)
        split.addWidget(detail)
        split.setSizes([355, 765])
        outer = W.QVBoxLayout(self)
        outer.addWidget(split)
        self.setStyleSheet("""
            QWidget { background: #f5f7fb; color: #17263b; font-size: 14px; }
            QLabel#title { font-size: 23px; font-weight: 700; }
            QTreeWidget, QLineEdit, QPlainTextEdit, QComboBox {
                background: white; border: 1px solid #cdd7e5; border-radius: 5px; padding: 5px; }
            QTreeWidget::item { min-height: 25px; }
            QTreeWidget::item:selected { background: #d9e8ff; color: #15365c; }
            QPushButton { background: white; border: 1px solid #aebed2; padding: 7px 12px; border-radius: 5px; }
            QPushButton#run { background: #205ab0; color: white; }
            QPushButton:disabled, QPushButton#run:disabled { background: #e4e9f1; color: #53647b; }
        """)
        self.populate()

    def populate(self):
        groups = {}
        first = None
        for script in project_scripts():
            chapter, *parts = script.relative_to(ROOT).parts
            if chapter not in groups:
                groups[chapter] = W.QTreeWidgetItem(self.tree, [CHAPTERS[chapter[0]]])
                groups[chapter].setFlags(C.Qt.ItemIsEnabled)
            item = W.QTreeWidgetItem(groups[chapter], [" / ".join(parts)])
            item.setData(0, C.Qt.UserRole, str(script))
            if first is None:
                first = item
        self.tree.expandAll()
        if first:
            self.tree.setCurrentItem(first)
        else:
            self.select_project(None)

    def filter_projects(self, query):
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            visible = False
            for n in range(group.childCount()):
                child = group.child(n)
                match = query.casefold().strip() in (group.text(0) + " " + child.text(0)).casefold()
                child.setHidden(not match)
                visible |= match
            group.setHidden(not visible)

    def select_project(self, item, *_):
        path = item.data(0, C.Qt.UserRole) if item else None
        self.script = Path(path) if path else None
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields = []
        if not self.script:
            self.run_button.setEnabled(False)
            self.folder.setEnabled(False)
            return
        path = self.script.relative_to(ROOT)
        self.title.setText(" / ".join(path.parts[1:]))
        self.folder.setEnabled(True)
        self.run_button.setEnabled(True)
        self.camera = "/".join(path.parts[1:]) in CAMERA_CASES and path.parts[0].startswith("7.")
        self.server = self.script.name == "server.py"
        self.mode = None
        self.ocr = None
        self.note.setText("Örnek, kendi klasöründeki kitap verileriyle çalışır.")
        if self.camera:
            self.note.setText("Bu örnek canlı kamera kullanır. Bitirmek için Durdur düğmesine bas.")
        if self.server:
            self.note.setText("Sunucu başladıktan sonra Yerel sayfayı aç düğmesini kullan.")
        fields = FIELDS.get(self.script.name, ())
        if self.script.name == "prediction.py":
            if path.parts[0].startswith("4."):
                fields = (("Yorum", "Ürünü beğendim, kargo hızlıydı."),)
            elif "1.Proje" in path.parts:
                fields = FIELDS["ConsolePrediction.py"]
        for label, default in fields:
            entry = W.QLineEdit(default)
            self.form.addRow(label, entry)
            self.fields.append(entry)
        if self.script.name == "app.py" and "Göz ile fare kontrolü" in path.parts:
            self.mode = W.QComboBox()
            self.mode.addItems(["Fotoğrafta göz tespiti", "Fotoğrafta kırpma skoru",
                                "Canlı kamera", "Canlı kamera ve fare kontrolü"])
            self.form.addRow("Çalışma modu", self.mode)
        if "2.Proje" in path.parts and self.script.name in {"bot.py", "prediction.py"}:
            self.mode = W.QComboBox()
            self.mode.addItems(["Olumlu yorum", "Olumsuz yorum"])
            self.form.addRow("Yorum türü", self.mode)
        if "plaka okuma" in path.parts:
            self.ocr = W.QCheckBox("Tesseract ile metni oku")
            self.form.addRow(self.ocr)
            self.note.setText("Plaka bölgesi önizlemede gösterilir. Metin okuma için Tesseract gerekir.")

    def run_project(self):
        if self.process.state() != C.QProcess.NotRunning or not self.script:
            return
        script = self.script
        args = []
        live = self.camera or self.server
        if self.mode is not None:
            mode = self.mode.currentIndex()
            if script.name == "app.py":
                args = [[], ["--blink"], ["--camera"], ["--camera", "--mouse"]][mode]
                live = mode >= 2
            else:
                args = ["--sentiment", str(1 - mode)]
        if live and not self.server:
            label = "kamerayı ve fare kontrolünü" if "--mouse" in args else "kamerayı"
            if W.QMessageBox.question(self, "Canlı örnek", f"Bu örnek {label} açacak. Başlatılsın mı?",
                                      W.QMessageBox.Yes | W.QMessageBox.No,
                                      W.QMessageBox.No) != W.QMessageBox.Yes:
                return
        values = [entry.text() for entry in self.fields]
        if script.name == "cumle.py":
            args, values = ["--", *values], []
        elif script.name == "kelime.py":
            args, values = [f"--word={values[0]}", f"--other={values[1]}"], []
        if self.ocr is not None and self.ocr.isChecked():
            args.append("--ocr")
        if script.name == "KNN.py":
            values = ["y", *values, "x"]
        self.pending_input = ("\n".join(values) + "\n").encode("utf-8") if values else b""
        self.preview_dir = ROOT / "outputs" / "arayuz" / uuid4().hex[:12]
        try:
            self.preview_dir.mkdir(parents=True)
        except OSError as error:
            self.status.setText(f"Çıktı klasörü açılamadı. {error}")
            return
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.previous_images = self.output_images()
        self.stopping = False
        self.log.clear()
        self.append_log(f"{script.relative_to(ROOT)}\n\n")
        self.images.clear()
        self.tabs.setCurrentIndex(0)
        self.status.setText("Başlatılıyor…")
        self.set_busy(True)
        env = C.QProcessEnvironment.systemEnvironment()
        if not live:
            env.insert("MPLBACKEND", "Agg")
            face = ROOT / "7. Computer Vision" / "Yüz işlemleri" / "mo.png"
            command = [str(ROOT / "dogrula.py"), "--worker", str(script),
                       str(self.preview_dir), str(face), "script", *args]
        else:
            command = [str(script), *args]
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(script.parent))
        # pythonw.exe may launch the GUI. Child scripts need standard streams.
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        self.process.start(str(python), ["-X", "utf8", "-u", *command])

    def send_initial_input(self):
        if self.pending_input:
            self.process.write(self.pending_input)
        self.process.closeWriteChannel()
        self.status.setText("Çalışıyor…")
        self.browser.setVisible(self.server)

    def read_output(self):
        data = bytes(self.process.readAllStandardOutput())
        self.append_log(self.decoder.decode(data))

    def append_log(self, text):
        self.log.moveCursor(G.QTextCursor.End)
        self.log.insertPlainText(text.replace("\r", "\n"))
        self.log.ensureCursorVisible()

    def set_busy(self, busy):
        for widget in (self.search, self.tree, self.run_button, *self.fields):
            widget.setEnabled(not busy)
        if self.mode is not None:
            self.mode.setEnabled(not busy)
        if self.ocr is not None:
            self.ocr.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if not busy:
            self.browser.hide()

    def process_error(self, error):
        if error == C.QProcess.FailedToStart:
            self.append_log(self.process.errorString())
            self.status.setText("Python süreci başlatılamadı.")
            self.set_busy(False)
            if self.closing:
                C.QTimer.singleShot(0, self.close)

    def finished(self, code, exit_status):
        self.read_output()
        self.append_log(self.decoder.decode(b"", final=True))
        if self.stopping:
            self.status.setText("Durduruldu.")
        elif exit_status == C.QProcess.NormalExit and code == 0:
            self.status.setText("Başarıyla tamamlandı.")
        else:
            self.status.setText(f"Proje hata ile kapandı (kod {code}). Ayrıntılar Çıktı sekmesinde.")
        saved_images = self.output_images()
        previews = sorted(self.preview_dir.glob("*.png"))
        previews.extend(path for path, stamp in sorted(saved_images.items())
                        if self.previous_images.get(path) != stamp)
        for path in previews:
            self.images.addItem(path.name, str(path))
        if previews and code == 0 and not self.stopping:
            self.tabs.setCurrentIndex(1)
        self.set_busy(False)
        if self.closing:
            C.QTimer.singleShot(0, self.close)

    def show_preview(self, index):
        data = self.images.itemData(index) if index >= 0 else None
        if not data:
            self.picture.clear()
            self.picture.setText("Görsel çıktı burada gösterilir.")
            return
        image = G.QPixmap(data)
        if not image.isNull():
            self.picture.setPixmap(image.scaled(self.picture.size(), C.Qt.KeepAspectRatio,
                                               C.Qt.SmoothTransformation))

    def output_images(self):
        images = {}
        for folder in (self.script.parent, self.script.parent / "outputs",
                       self.script.parent / "Images"):
            for path in folder.glob("*"):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    try:
                        images[path] = path.stat().st_mtime_ns
                    except OSError:
                        continue
        return images

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_preview(self.images.currentIndex())

    def open_folder(self):
        if self.script:
            G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(str(self.script.parent)))

    def stop(self):
        if self.process.state() != C.QProcess.NotRunning:
            self.stopping = True
            self.process.kill()

    def closeEvent(self, event):
        if self.process.state() != C.QProcess.NotRunning:
            self.closing = True
            self.stop()
            event.ignore()
        else:
            event.accept()


def main():
    app = W.QApplication(sys.argv)
    window = Launcher()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
