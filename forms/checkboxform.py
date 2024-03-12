from forms.checkboxform_ui import Ui_DialogCheckbox
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QDialog, QFileDialog, QTreeWidgetItem

class CheckboxForm(QDialog):
    def __init__(self, parent:'MainForm', elements:list):
        super().__init__(parent)
        self.ui = Ui_DialogCheckbox()
        self.ui.setupUi(self)
        # наименования элементов для выбора
        self.elements = elements
        self.checked_elements = []
        self.setWindowTitle('Выбор элементов')
        self.ui.checkBox_cubes.stateChanged.connect(self.selectAll)
        self.ui.pushButtonOk.clicked.connect(self.func_ok)
        self.fill_tree()

    def fill_tree(self):
        for el in self.elements:
            cube_level = QTreeWidgetItem()
            cube_level.setText(0, str(el))
            cube_level.setCheckState(0, Qt.Unchecked)
            self.ui.treeWidget.addTopLevelItem(cube_level)
    
    def selectAll(self):
        """Если нажали на чекбокс 'Выбрать все'"""
        if self.ui.checkBox_cubes.isChecked():
            check_state = Qt.Checked
        else:
            check_state = Qt.Unchecked
        for idx in range(self.ui.treeWidget.topLevelItemCount()):
            item = self.ui.treeWidget.topLevelItem(idx)
            item.setCheckState(0, check_state)
    
    def func_ok(self):
        self.checked_elements = []
        for idx in range(self.ui.treeWidget.topLevelItemCount()):
            item = self.ui.treeWidget.topLevelItem(idx)
            if item.checkState(0) == Qt.Checked:
                self.checked_elements.append(item.text(0))
        self.close()