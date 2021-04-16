import sys
import math
import numpy as np
import numpy.random as nr
from scipy.stats import weibull_min
from scipy.special import gamma
from os import environ
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QTableWidgetItem


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


FACTORS_NUMBER = 4
MATR_SIZE = 2 ** FACTORS_NUMBER
MOD_NUMBER = 1


class GaussDistribution:
    def __init__(self, m: float, sigma: float):
        self._m = m
        self._sigma = sigma

    def generate(self):
        return nr.normal(self._m, self._sigma)


class WeibullDistribution:
    def __init__(self, k: float, lambd: float):
        self._k = k
        self._lambd = lambd

    def generate(self):
        return weibull_min.rvs(self._k, loc=0, scale=self._lambd)


class Generator:
    def __init__(self, generator):
        self._generator = generator
        self._receivers = set()
        self.intensity = 0
        self.request_count = 0

    def add_receiver(self, receiver):
        self._receivers.add(receiver)

    def remove_receiver(self, receiver):
        try:
            self._receivers.remove(receiver)
        except KeyError:
            pass

    def next_time(self):
        new_time = self._generator.generate()
        self.intensity += new_time
        self.request_count += 1
        return new_time

    def emit_request(self):
        for receiver in self._receivers:
            receiver.receive_request()

    def get_avg_intensity(self):
        return 1 / (self.intensity / self.request_count)


class Processor(Generator):
    def __init__(self, generator, reenter_probability=0):
        super().__init__(generator)
        self._current_queue_size = 0
        self._max_queue_size = 0
        self._processed_requests = 0
        self._reenter_probability = reenter_probability
        self._reentered_requests = 0

    def process(self):
        if self._current_queue_size > 0:
            self._processed_requests += 1
            self._current_queue_size -= 1
            self.emit_request()
            if nr.random_sample() <= self._reenter_probability:
                self._reentered_requests += 1
                self._processed_requests -= 1
                self.receive_request()

    def receive_request(self):
        self._current_queue_size += 1
        if self._current_queue_size > self._max_queue_size:
            self._max_queue_size = self._current_queue_size

    @property
    def processed_requests(self):
        return self._processed_requests

    @property
    def max_queue_size(self):
        return self._max_queue_size

    @property
    def current_queue_size(self):
        return self._current_queue_size

    @property
    def reentered_requests(self):
        return self._reentered_requests



class Modeller:
    def __init__(self, uniform_a, uniform_b, weibull_a, weibull_lamb,):
        self._generator = Generator(GaussDistribution(uniform_a, uniform_b))
        self._processor = Processor(WeibullDistribution(weibull_a, weibull_lamb))
        self._generator.add_receiver(self._processor)

    def event_based_modelling(self, end_time):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period + processor.next_time()

        start_times = [gen_period]
        end_times = [proc_period]

        cur_time = 0
        queue = 0

        while cur_time < end_time:
            if gen_period <= proc_period:
                generator.emit_request()
                gen_period += generator.next_time()
                cur_time = gen_period
                start_times.append(cur_time)
            if gen_period >= proc_period:
                processor.process()
                if processor.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()
                cur_time = proc_period
                end_times.append(cur_time)

        avg_wait_time = 0
        request_count = min(len(end_times), len(start_times))

        tmp = []
        for i in range(request_count):
            avg_wait_time += end_times[i] - start_times[i]
            tmp.append(end_times[i] - start_times[i])

        if request_count > 0:
            avg_wait_time /= request_count

        actual_lamb = self._generator.get_avg_intensity()
        actual_mu = self._processor.get_avg_intensity()
        ro = actual_lamb / actual_mu
        print("actual_ro", actual_lamb, actual_mu)

        return ro, avg_wait_time

    def time_based_modelling(self, request_count, dt):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period + processor.next_time()
        current_time = 0

        while processor.processed_requests < request_count:
            if gen_period <= current_time:
                generator.emit_request()
                gen_period += generator.next_time()
            if current_time >= proc_period:
                processor.process()
                if processor.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()
            current_time += dt

        return (processor.processed_requests, processor.reentered_requests,
                processor.max_queue_size, round(current_time, 3))


class Experiment():
    def __init__(self, gen, proc, time):
        self.min_gen_int = gen[0]
        self.max_gen_int = gen[1]
        self.min_gen_var = gen[2]
        self.max_gen_var = gen[3]

        self.min_proc_int = proc[0]
        self.max_proc_int = proc[1]
        self.min_proc_var = proc[2]
        self.max_proc_var = proc[3]

        self.time = time
        self.coefs = []
        self.table = []

    def get_matrix(self):
        matrix = [[0 for i in range(MATR_SIZE)] for i in range(MATR_SIZE)]

        for i in range(MATR_SIZE):
            for j in range(1, FACTORS_NUMBER + 1):
                if i // (2 ** (FACTORS_NUMBER - j)) % 2 == 1:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = -1

            matrix[i][0] = 1

            matrix[i][5] = matrix[i][1] * matrix[i][2]
            matrix[i][6] = matrix[i][1] * matrix[i][3]
            matrix[i][7] = matrix[i][1] * matrix[i][4]
            matrix[i][8] = matrix[i][2] * matrix[i][3]
            matrix[i][9] = matrix[i][2] * matrix[i][4]
            matrix[i][10] = matrix[i][3] * matrix[i][4]

            matrix[i][11] = matrix[i][1] * matrix[i][2] * matrix[i][3]
            matrix[i][12] = matrix[i][1] * matrix[i][2] * matrix[i][4]
            matrix[i][13] = matrix[i][1] * matrix[i][3] * matrix[i][4]
            matrix[i][14] = matrix[i][2] * matrix[i][3] * matrix[i][4]

            matrix[i][15] = matrix[i][1] * matrix[i][2] * matrix[i][3] * matrix[i][4]

        return matrix

    def calc_xmat(self, plan):
        transposed = np.transpose(plan)
        mat = np.matmul(transposed, np.array(plan))
        mat = np.linalg.inv(mat)
        mat = np.matmul(mat, transposed)
        return mat.tolist()

    def linear(self, b, x):
        res = 0
        linlen = int(np.log2(len(b))) + 1
        for i in range(linlen):
            res += b[i] * x[i]
        return res

    def nonlinear(self, b, x):
        res = 0
        for i in range(len(b)):
            res += b[i] * x[i]
        return res

    def expand_plan(self, plan, y, xmat):
        b = list()
        for i in range(len(xmat)):
            b_cur = 0
            for j in range(len(xmat[i])):
                b_cur += xmat[i][j] * y[j]
            b.append(b_cur)

        ylin = list()
        ynlin = list()
        for i in range(len(plan)):
            ylin.append(self.linear(b, plan[i]))
            ynlin.append(self.nonlinear(b, plan[i]))

        for i in range(len(plan)):
            plan[i].append(y[i])
            plan[i].append(ylin[i])
            plan[i].append(ynlin[i])
            plan[i].append(abs(y[i] - ylin[i]))
            plan[i].append(abs(y[i] - ynlin[i]))

        return plan, b

    def scale_factor(self, x, realmin, realmax, xmin=-1, xmax=1):
        return realmin + (realmax - realmin) * (x - xmin) / (xmax - xmin)

    def param_convert(self, gen_int, gen_var, pm_int, pm_var):
        a = 1 / gen_int
        b = 1 / gen_var

        weib_a = (pm_int * pm_var) ** (-1.086)
        weib_lamb = 1 / (pm_int * gamma(1 + 1 / weib_a))
        return a, b, weib_a, weib_lamb

    def calculate(self):
        matrix = self.get_matrix()

        xmat = self.calc_xmat(matrix)

        y = list()

        for exp in matrix:
            gen_int = self.scale_factor(exp[1], self.min_gen_int, self.max_gen_int)
            gen_var = self.scale_factor(exp[2], self.min_gen_var, self.max_gen_var)
            pm_int = self.scale_factor(exp[3], self.min_proc_int, self.max_proc_int)
            pm_var = self.scale_factor(exp[4], self.min_proc_var, self.max_proc_var)

            a, b, weib_a, weib_lamb = self.param_convert(gen_int, gen_var, pm_int, pm_var)

            exp_res = 0
            for i in range(MOD_NUMBER):
                model = Modeller(a, b, weib_a, weib_lamb)
                ro, avg_wait_time = model.event_based_modelling(self.time)
                exp_res += avg_wait_time
            exp_res /= MOD_NUMBER

            y.append(exp_res)

        plan, self.coefs = self.expand_plan(matrix, y, xmat)
        return plan, self.coefs

    def check(self, gen_int, gen_var, pm_int, pm_var):

        exp_res = 0
        for i in range(MOD_NUMBER):
            new_gen_int = self.scale_factor(gen_int, self.min_gen_int, self.max_gen_int)
            new_gen_var = self.scale_factor(gen_var, self.min_gen_var, self.max_gen_var)
            new_proc_int = self.scale_factor(pm_int, self.min_proc_int, self.max_proc_int)
            new_proc_var = self.scale_factor(pm_var, self.min_proc_var, self.max_proc_var)

            a, b, weib_a, weib_lamb = self.param_convert(new_gen_int, new_gen_var, new_proc_int, new_proc_var)
            model = Modeller(a, b, weib_a, weib_lamb)
            ro, avg_wait_time = model.event_based_modelling(self.time)
            exp_res += avg_wait_time

        exp_res /= MOD_NUMBER

        lin_res = self.coefs[0] + self.coefs[1] * gen_int + self.coefs[2] * gen_var + self.coefs[3] * pm_int + self.coefs[4] * pm_var
        nonlin_res = self.coefs[0] + self.coefs[1] * gen_int + self.coefs[2] * gen_var + self.coefs[3] * pm_int + self.coefs[4] * pm_var + \
                     self.coefs[5] * gen_int * gen_var + self.coefs[6] * gen_int * pm_int + self.coefs[7] * gen_int * pm_var + \
                     self.coefs[8] * gen_var * pm_int + self.coefs[9] * gen_var * pm_var + self.coefs[10] * pm_int * pm_var + \
                     self.coefs[11] * gen_int * gen_var * pm_int + self.coefs[12] * gen_int * gen_var * pm_var + \
                     self.coefs[13] * gen_int * pm_int * pm_var + self.coefs[14] * gen_var * gen_var * pm_var + \
                     self.coefs[15] * gen_int * gen_var * pm_int * pm_var

        return [gen_int, gen_var, pm_int, pm_var, exp_res, lin_res, nonlin_res]

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("window.ui", self)
        self.experiment = None
        self.Table_position = 1

    @pyqtSlot(name='on_calcButton_clicked')
    def _parse_parameters(self):
        try:
            ui = self.ui

            min_gen_int = float(ui.line_edit_min_gen_int.text())
            max_gen_int = float(ui.line_edit_max_gen_int.text())
            min_gen_var = float(ui.line_edit_min_gen_var.text())
            max_gen_var = float(ui.line_edit_max_gen_var.text())
            generator = [min_gen_int, max_gen_int, min_gen_var, max_gen_var]

            min_proc_int = float(ui.line_edit_min_pm_int.text())
            max_proc_int = float(ui.line_edit_max_pm_int.text())
            min_proc_var = float(ui.line_edit_min_pm_var.text())
            max_proc_var = float(ui.line_edit_max_pm_var.text())
            processor = [min_proc_int, max_proc_int, min_proc_var, max_proc_var]
            if min_gen_int < 0 or max_gen_int < 0 or min_gen_var < 0 or max_gen_var < 0 or \
                    min_proc_int < 0 or max_proc_int < 0 or min_proc_var < 0 or max_proc_var < 0:
                raise ValueError('Интенсивности и дисперсии интенсивностей должны быть > 0')

            time = int(ui.line_edit_time.text())
            if time <= 0:
                raise ValueError('Необходимо время моделирования > 0')

            self.experiment = Experiment(generator, processor, time)
            table, regr = self.experiment.calculate()

            self._show_results(table, regr)
        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка входных данных!\n' + str(e))
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', e)

    def set_value(self, line, column, format, value):
        item = QTableWidgetItem(format % value)
        item.setTextAlignment(Qt.AlignRight)
        self.ui.table.setItem(line, column, item)

    def _show_results(self, table, regr):
        ui = self.ui

        accuracy = 3

        lin_regr = str(round(regr[0], accuracy)) + " + " + str(round(regr[1], accuracy)) + "x1 + " + str(
                    round(regr[2], accuracy)) + "x2 + " + str(round(regr[3], accuracy)) + "x3 + " + \
                   str(round(regr[4], accuracy)) + "x4"
        lin_regr = lin_regr.replace("+ -", "- ")

        nonlin_regr = lin_regr + " + " + str(round(regr[5], accuracy)) + "x1x2 + " + str(round(regr[6], accuracy)) + \
                      "x1x3 + " + str(round(regr[7], accuracy)) + "x1x4 + " + str(round(regr[8], accuracy)) + "x2x3 +" + \
                      str(round(regr[9], accuracy)) + "x2x4 + " + str(round(regr[10], accuracy)) + "x3x4 + " + \
                      str(round(regr[11], accuracy)) + "x1x2x3 + " + str(round(regr[12], accuracy)) + "x1x2x4 + " + \
                      str(round(regr[13], accuracy)) + "x1x3x4 + " + str(round(regr[14], accuracy)) + "x2x3x4 + " + \
                      str(round(regr[15], accuracy)) + "x1x2x3x4"

        nonlin_regr = nonlin_regr.replace("+ -", "- ")

        ui.line_edit_lin_res.setText(lin_regr)
        ui.line_edit_nonlin_res.setText(nonlin_regr)

        for i in range(len(table)):
            ui.table.setRowCount(ui.Table_position + 1)
            table_len = len(table[i])
            for j in range(table_len + 1):
                if j == 0:
                    self.set_value(ui.Table_position, 0, '%d', ui.Table_position);
                elif j < table_len - 4:
                    self.set_value(ui.Table_position, j, '%d', table[i][j - 1]);
                else:
                    self.set_value(ui.Table_position, j, '%.3f', table[i][j - 1])
            ui.Table_position += 1

    @pyqtSlot(name='on_checkButton_clicked')
    def _parse_check_parameters(self):
        try:
            ui = self.ui

            gen_int = float(ui.line_edit_gen_int.text())
            gen_var = float(ui.line_edit_gen_var.text())
            proc_int = float(ui.line_edit_pm_int.text())
            proc_var = float(ui.line_edit_pm_var.text())

            if abs(gen_int) > 1 or abs(gen_var) > 1 or abs(proc_int) > 1 or abs(proc_var) > 1:
                raise ValueError('Координаты точки должны находится в диапазоне [-1; 1]')

            time = int(ui.line_edit_time.text())
            if time <= 0:
                raise ValueError('Необходимо время моделирования > 0')

            res = self.experiment.check(gen_int, gen_var, proc_int, proc_var)

            self._show_table(res)
        except ValueError as e:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка входных данных!\n' + str(e))
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', e)

    def _show_table(self, res):
        ui = self.ui
        gen_int = res[0]
        gen_var = res[1]
        proc_int = res[2]
        proc_var = res[3]
        exp_res = res[4]
        lin_res = res[5]
        nonlin_res = res[6]
        res = [1, gen_int, gen_var, proc_int, proc_var, gen_int * gen_var, gen_int * proc_int, proc_int * proc_var,
               gen_var * proc_int, gen_var * proc_var, proc_int * proc_var, gen_int * gen_var * proc_int,
               gen_int * gen_var * proc_var, gen_int * proc_int * proc_var, gen_var * proc_int * proc_var,
               gen_int * gen_var * proc_int * proc_var, exp_res, lin_res, nonlin_res, exp_res - lin_res,
               exp_res - nonlin_res]

        ui.table.setRowCount(ui.Table_position + 1)
        table_len = len(res)
        for j in range(table_len + 1):
            if j == 0:
                self.set_value(ui.Table_position, 0, '%d', ui.Table_position)
            elif j < table_len - 4:
                self.set_value(ui.Table_position, j, '%g', res[j - 1])
            else:
                self.set_value(ui.Table_position, j, '%.3f', abs(res[j - 1]))
        ui.Table_position += 1


def qt_app():
    suppress_qt_warnings()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
