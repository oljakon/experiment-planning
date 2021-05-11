from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLineEdit, QTableWidgetItem
import sys
import numpy as np
from scipy.special import gamma

import modeller


def calculate_params(la1, dla1, la2, dla2, mu, dmu):
    mT11 = 1 / la1
    dT11 = (1 / (la1 - dla1) - 1 / (la1 + dla1)) / 2

    mT12 = 1 / la2
    dT12 = (1 / (la2 - dla2) - 1 / (la2 + dla2)) / 2

    try:
        mT2 = (mu * dmu) ** (-1.086)
    except ZeroDivisionError:
        mT2 = 0.0
    dT2 = 1 / (mu * gamma(1 + 1 / mT2))

    return mT11, dT11, mT12, dT12, mT2, dT2


def process_matrixes(initialMatrix):
    levelMatrix = [[0.0 for j in range(len(initialMatrix[0]))] for i in range(len(initialMatrix))]

    for i in range(len(levelMatrix)):
        for j in range(len(levelMatrix[0])):
            try:
                levelMatrix[i][j] = float(initialMatrix[i][j])
            except:
                levelMatrix[i][j] = 0.0

    planningMatrix = np.matrix(list(map(lambda row: row[:64], levelMatrix.copy()[:-1])))
    checkVector = np.array(levelMatrix.copy()[-1][:64])
    transposedPlanningMatrix = planningMatrix.transpose()

    return np.linalg.inv(
        transposedPlanningMatrix * planningMatrix) * transposedPlanningMatrix, planningMatrix, checkVector


def convert_value_to_factor(min, max, value):
    return (value - (max + min) / 2.0) / ((max - min) / 2.0)


def convert_factor_to_value(min, max, factor):
    return factor * ((max - min) / 2.0) + (max + min) / 2.0


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("mainwindow.ui", self)

        self.la1 = 0
        self.dla1 = 1
        self.la2 = 0
        self.dla2 = 1
        self.mu = 0
        self.dmu = 1
        self.tmax = 300

        self.read_params()

        self.init_table()
        self.init_table2()

        self.set_free_point()

    @pyqtSlot(name='on_calcButton_clicked')
    def on_calculate_model(self):
        self.calculate_ffe()
        self.calculate_pfe()

    def calculate_ffe(self):
        tableWidget = self.ui.tableWidget

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()

        planningTable = [[tableWidget.item(i, j).text() for j in range(cols)] for i in range(rows)]

        coefMatrix, planningMatrix, checkVector = process_matrixes(planningTable)

        factorMatrix = np.matrix(list(map(lambda row: row[1:7], planningTable.copy())))


        Y = [0 for i in range(65)]

        for i in range(len(factorMatrix.tolist())):
            la1 = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla1 = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            la2 = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dla2 = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))
            mu = convert_factor_to_value(Xmin[4], Xmax[4], float(factorMatrix.item((i, 4))))
            dmu = convert_factor_to_value(Xmin[5], Xmax[5], float(factorMatrix.item((i, 5))))

            mT11, dT11, mT12, dT12, mT2, dT2 = calculate_params(la1, dla1, la2, dla2, mu, dmu)

            model = modeller.Model([mT11, mT12], [dT11, dT12], mT2, dT2, 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modelling(100, 0.001)

            Y[i] = avg_queue_time
            tableWidget.setItem(i, 64, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])

        B = (coefMatrix @ Y).tolist()[0]

        y0 = ("y = %.6fb0 + %.6fb1 + %.6fb2 + %.6fb3 + %.6fb4 + %.6fb5 + %.6fb6" % \
              (B[0], B[1], B[2], B[3], B[4], B[5], B[6]))

        y0 = str(y0)
        y0 = y0.replace("+ -", "- ")
        self.equ0.setText(y0)

        y1 = ("y = %.6fb0 + %.6fb1 + %.6fb2 + %.6fb3 + %.6fb4 + %.6fb5 + %.6fb6 + %.6fb12 + %.6fb13 + %.6fb14 "
              "+ %.6fb15 + %.6fb16 + %.6fb23 + %.6fb24 + %.6fb25 + %.6fb26 + %.6fb34 + %.6fb35 + %.6fb36 + %.6fb45 "
              "+ %.6fb46 + %.6fb56 + %.6fb123 + %.6fb124 + %.6f125 + %.6fb126 + %.6fb134 + %.6fb135 + %.6fb136 "
              "+ %.6fb145 + %.6fb146 + %.6fb156 + %.6fb234 + %.6fb235 + %.6fb236 + %.6fb245 + %.6fb246 + %.6fb256 "
              "+ %.6fb345 + %.6fb346 + %.6fb356 + %.6fb456 + %.6fb1234 + %.6fb1235 + %.6fb1236 + %.6fb1245 + %.6fb1246 "
              "+ %.6fb1256 + %.6fb1345 + %.6fb1346 + %.6fb1356 + %.6fb1456 + %.6fb2345 + %.6fb2346+ %.6fb2356 "
              "+ %.6fb2456 + %.6fb3456 + %.6fb12345 + %.6fb12346 + %.6fb12356 + %.6fb12456 + %.6fb13456 + %.6fb23456 "
              "+ %.6fb123456" % \
              (B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9], B[10], B[11], B[12], B[13], B[14], B[15],
               B[16], B[17], B[18], B[19], B[20], B[21], B[22], B[23], B[24], B[25], B[26], B[27], B[28], B[29],
               B[30], B[31], B[32], B[33], B[34], B[35], B[36], B[37], B[38], B[39], B[40], B[41], B[42], B[43],
               B[44], B[45], B[46], B[47], B[48], B[49], B[50], B[51], B[52], B[53], B[54], B[55], B[56], B[57],
               B[58], B[59], B[60], B[61], B[62], B[63]))

        y1 = str(y1)
        y1 = y1.replace("+ -", "- ")
        self.equ1.setText(y1)
        Yl = np.array(list(map(lambda row: row[:7], planningMatrix.tolist() + [checkVector.tolist()]))) @ np.array(
            B[:7])
        Ypn = np.array(planningMatrix.tolist() + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 65, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 66, QTableWidgetItem(str(round(Ypn.tolist()[i], 4))))
            tableWidget.setItem(i, 67, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 68, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Ypn.tolist()[i], 6), 6)))))

    def calculate_pfe(self):
        tableWidget = self.ui.tableWidget2

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()

        planningTable = [[float(tableWidget.item(i, j).text()) for j in range(64)] for i in range(rows)]
        factorMatrix = np.matrix(list(map(lambda row: row[1:7], planningTable.copy())))
        checkVector = np.array(planningTable.copy()[-1][:64])

        Y = [0 for i in range(9)]

        for i in range(len(factorMatrix.tolist())):
            la1 = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla1 = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            la2 = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dla2 = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))
            mu = convert_factor_to_value(Xmin[4], Xmax[4], float(factorMatrix.item((i, 4))))
            dmu = convert_factor_to_value(Xmin[5], Xmax[5], float(factorMatrix.item((i, 5))))

            mT11, dT11, mT12, dT12, mT2, dT2 = calculate_params(la1, dla1, la2, dla2, mu, dmu)

            model = modeller.Model([mT11, mT12], [dT11, dT12], mT2, dT2, 2, 1, 0)

            avg_queue_size, avg_queue_time, processed_requests = model.time_based_modelling(100, 0.001)

            Y[i] = avg_queue_time
            tableWidget.setItem(i, 64, QTableWidgetItem(str(round(avg_queue_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])

        B = [np.array([float(planningTable[i][k]) / len(Y) for i in range(len(Y))]) @ Y for k in range(64)]

        Yl = np.array(list(map(lambda row: row[:7], planningTable + [checkVector.tolist()]))) @ np.array(
            B[:7])

        Bl = B[:7] + [0 for i in range(7, len(B))]

        Bpn = B.copy()

        for i in range(0, 64):
            B[i] = B[i] / self.count_eq_rows(planningTable, i, len(planningTable[0]))

        y2 = ("y = %.6fb0 + %.6fb1 + %.6fb2 + %.6fb3 + %.6fb4 + %.6fb5 + %.6fb6" % \
              (B[0], B[1], B[2], B[3], B[4], B[5], B[6]))

        y2 = str(y2)
        y2 = y2.replace("+ -", "- ")
        self.equ2.setText(y2)

        y3 = ("y = %.6fb0 + %.6fb1 + %.6fb2 + %.6fb3 + %.6fb4 + %.6fb5 + %.6fb6 + %.6fb12 + %.6fb13 + %.6fb14 "
              "+ %.6fb15 + %.6fb16 + %.6fb23 + %.6fb24 + %.6fb25 + %.6fb26 + %.6fb34 + %.6fb35 + %.6fb36 + %.6fb45 "
              "+ %.6fb46 + %.6fb56 + %.6fb123 + %.6fb124 + %.6f125 + %.6fb126 + %.6fb134 + %.6fb135 + %.6fb136 "
              "+ %.6fb145 + %.6fb146 + %.6fb156 + %.6fb234 + %.6fb235 + %.6fb236 + %.6fb245 + %.6fb246 + %.6fb256 "
              "+ %.6fb345 + %.6fb346 + %.6fb356 + %.6fb456 + %.6fb1234 + %.6fb1235 + %.6fb1236 + %.6fb1245 + %.6fb1246 "
              "+ %.6fb1256 + %.6fb1345 + %.6fb1346 + %.6fb1356 + %.6fb1456 + %.6fb2345 + %.6fb2346+ %.6fb2356 "
              "+ %.6fb2456 + %.6fb3456 + %.6fb12345 + %.6fb12346 + %.6fb12356 + %.6fb12456 + %.6fb13456 + %.6fb23456 "
              "+ %.6fb123456" % \
              (B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9], B[10], B[11], B[12], B[13], B[14], B[15],
               B[16], B[17], B[18], B[19], B[20], B[21], B[22], B[23], B[24], B[25], B[26], B[27], B[28], B[29],
               B[30], B[31], B[32], B[33], B[34], B[35], B[36], B[37], B[38], B[39], B[40], B[41], B[42], B[43],
               B[44], B[45], B[46], B[47], B[48], B[49], B[50], B[51], B[52], B[53], B[54], B[55], B[56], B[57],
               B[58], B[59], B[60], B[61], B[62], B[63]))

        y3 = str(y3)
        y3 = y3.replace("+ -", "- ")
        self.equ3.setText(y3)

        for i in range(0, 22):
            Bpn[i] = Bpn[i] / self.count_eq_rows(planningTable, i, 22)
        for i in range(22, 64):
            Bpn[i] = 0

        Ypn = np.array(planningTable + [checkVector.tolist()]) @ np.array(Bpn)

        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 65, QTableWidgetItem(str(round(Yl.tolist()[i], 4))))
            tableWidget.setItem(i, 66, QTableWidgetItem(str(round(Ypn.tolist()[i], 4))))
            tableWidget.setItem(i, 67, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yl.tolist()[i], 6), 6)))))
            tableWidget.setItem(i, 68, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Ypn.tolist()[i], 6), 6)))))

    def count_eq_rows(self, plTable, i, N):
        count = 0

        for j in range(N):
            eq = True
            for k in range(len(plTable) - 1):
                eq = eq and (plTable[k][j] == plTable[k][i])
            if eq:
                count += 1

        return count

    def read_params(self):
        tmax = 300

        la1 = float(self.x1.text())
        dla1 = float(self.x2.text())
        la2 = float(self.x3.text())
        dla2 = float(self.x4.text())
        mu = float(self.x5.text())
        dmu = float(self.x6.text())

        Xmin, Xmax = self.read_model_params()
        la1 = convert_factor_to_value(Xmin[0], Xmax[0], la1)
        dla1 = convert_factor_to_value(Xmin[1], Xmax[1], dla1)
        la2 = convert_factor_to_value(Xmin[2], Xmax[2], la2)
        dla2 = convert_factor_to_value(Xmin[3], Xmax[3], dla2)
        mu = convert_factor_to_value(Xmin[4], Xmax[4], mu)
        dmu = convert_factor_to_value(Xmin[5], Xmax[5], dmu)

        self.la1 = la1
        self.la2 = la2
        self.mu = mu
        self.dla1 = dla1
        self.dla2 = dla2
        self.dmu = dmu
        self.tmax = tmax


    def read_model_params(self):
        Xmin = [0, 0, 0, 0, 0, 0]
        Xmax = [0, 0, 0, 0, 0, 0]

        Xmin[0] = Xmin[2] = float(self.gen_int_min.text())
        Xmax[0] = Xmax[2] = float(self.gen_int_max.text())
        Xmin[1] = Xmin[3] = float(self.gen_range_min.text())
        Xmax[1] = Xmax[3] = float(self.gen_range_max.text())
        Xmin[4] = float(self.oa_int_min.text())
        Xmax[4] = float(self.oa_int_max.text())
        Xmin[5] = float(self.oa_range_min.text())
        Xmax[5] = float(self.oa_range_max.text())

        return Xmin, Xmax

    def set_free_point(self):
        tableWidget = self.ui.tableWidget
        tableWidget2 = self.ui.tableWidget2

        Xmin, Xmax = self.read_model_params()
        x1 = convert_value_to_factor(Xmin[0], Xmax[0], self.la1)
        x2 = convert_value_to_factor(Xmin[1], Xmax[1], self.dla1)
        x3 = convert_value_to_factor(Xmin[2], Xmax[2], self.la2)
        x4 = convert_value_to_factor(Xmin[3], Xmax[3], self.dla2)
        x5 = convert_value_to_factor(Xmin[4], Xmax[4], self.mu)
        x6 = convert_value_to_factor(Xmin[5], Xmax[5], self.dmu)
        x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

        for i in range(64):
            tableWidget.setItem(64, i, QTableWidgetItem(str(round(x[i], 4))))

        x4 = x1 * x2
        x5 = x1 * x3
        x6 = x2 * x3

        x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

        for i in range(64):
            tableWidget2.setItem(8, i, QTableWidgetItem(str(x[i])))

    def set_b_table(self, B, table, row):
        for i in range(len(B)):
            table.setItem(row, i, QTableWidgetItem(str(round(B[i], 7))))

    def init_table(self):
        table = self.ui.tableWidget

        for i in range(table.rowCount() - 1):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = int(table.item(i, 4).text())
            x5 = int(table.item(i, 5).text())
            x6 = int(table.item(i, 6).text())

            x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

            for k in range(7, 64):
                table.setItem(i, k, QTableWidgetItem(str(x[k])))

    def init_table2(self):
        table = self.ui.tableWidget2

        for i in range(table.rowCount() - 1):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = x1 * x2
            x5 = x1 * x3
            x6 = x2 * x3

            table.setItem(i, 4, QTableWidgetItem(str(x4)))
            table.setItem(i, 5, QTableWidgetItem(str(x5)))
            table.setItem(i, 6, QTableWidgetItem(str(x6)))

            x = self.get_factor_array(x1, x2, x3, x4, x5, x6)

            for k in range(7, 64):
                table.setItem(i, k, QTableWidgetItem(str(x[k])))

    def get_factor_array(self, x1, x2, x3, x4, x5, x6):
        return [
            1,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x1 * x5,
            x1 * x6,
            x2 * x3,
            x2 * x4,
            x2 * x5,
            x2 * x6,
            x3 * x4,
            x3 * x5,
            x3 * x6,
            x4 * x5,
            x4 * x6,
            x5 * x6,
            x1 * x2 * x3,
            x1 * x2 * x4,
            x1 * x2 * x5,
            x1 * x2 * x6,
            x1 * x3 * x4,
            x1 * x3 * x5,
            x1 * x3 * x6,
            x1 * x4 * x5,
            x1 * x4 * x6,
            x1 * x5 * x6,
            x2 * x3 * x4,
            x2 * x3 * x5,
            x2 * x3 * x6,
            x2 * x4 * x5,
            x2 * x4 * x6,
            x2 * x5 * x6,
            x3 * x4 * x5,
            x3 * x4 * x6,
            x3 * x5 * x6,
            x4 * x5 * x6,
            x1 * x2 * x3 * x4,
            x1 * x2 * x3 * x5,
            x1 * x2 * x3 * x6,
            x1 * x2 * x4 * x5,
            x1 * x2 * x4 * x6,
            x1 * x2 * x5 * x6,
            x1 * x3 * x4 * x5,
            x1 * x3 * x4 * x6,
            x1 * x3 * x5 * x6,
            x1 * x4 * x5 * x6,
            x2 * x3 * x4 * x5,
            x2 * x3 * x4 * x6,
            x2 * x3 * x5 * x6,
            x2 * x4 * x5 * x6,
            x3 * x4 * x5 * x6,
            x1 * x2 * x3 * x4 * x5,
            x1 * x2 * x3 * x4 * x6,
            x1 * x2 * x3 * x5 * x6,
            x1 * x2 * x4 * x5 * x6,
            x1 * x3 * x4 * x5 * x6,
            x2 * x3 * x4 * x5 * x6,
            x1 * x2 * x3 * x4 * x5 * x6,
        ]


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
