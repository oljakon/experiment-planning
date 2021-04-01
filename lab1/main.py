import numpy.random as nr
from scipy.stats import weibull_min
from scipy.special import gamma
import sys

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from mainwindow import Ui_MainWindow


class GaussDistribution:
    def __init__(self, m: float, sigma: float):
        self._m = m
        self._sigma = sigma

    def generation_time(self):
        return nr.normal(self._m, self._sigma)


class WeibullDistribution:
    def __init__(self, k: float, lambd: float):
        self._k = k
        self._lambd = lambd

    def generation_time(self):
        return weibull_min.rvs(self._k, loc=0, scale=self._lambd)


class RequestGenerator:
    def __init__(self, generator):
        self._generator = generator
        self._receivers = set()
        self.time_periods = []

    def add_receiver(self, receiver):
        self._receivers.add(receiver)

    def remove_receiver(self, receiver):
        try:
            self._receivers.remove(receiver)
        except KeyError:
            pass

    def next_time_period(self):
        time = self._generator.generation_time()
        self.time_periods.append(time)
        return time

    def emit_request(self):
        for receiver in self._receivers:
            receiver.receive_request()


class RequestProcessor:
    def __init__(self, generator, len_queue=0, reenter_probability=0):
        self._generator = generator
        self._current_queue_size = 0
        self._max_queue_size = 0
        self._processed_requests = 0
        self._reenter_probability = reenter_probability
        self._reentered_requests = 0
        self._len_queue = len_queue
        self._num_lost_requests = 0
        self.time_periods = []

    @property
    def processed_requests(self):
        return self._processed_requests

    @property
    def lost_requests(self):
        return self._num_lost_requests

    @property
    def max_queue_size(self):
        return self._max_queue_size

    @property
    def current_queue_size(self):
        return self._current_queue_size

    @property
    def reentered_requests(self):
        return self._reentered_requests

    def process(self):
        if self._current_queue_size > 0:
            time_processed_request.append(current_time)
            self._processed_requests += 1
            self._current_queue_size -= 1

    def receive_request(self):
        self._current_queue_size += 1
        if self._current_queue_size > self._max_queue_size:
            self._max_queue_size += 1

    def next_time_period(self):
        time = self._generator.generation_time()
        self.time_periods.append(time)
        return time


current_time = 0
time_processed_request = []


class Modeller:
    def __init__(self, generator, processor):
        self._generator = generator
        self._processor = processor
        self._generator.add_receiver(self._processor)

    def time_based_modelling(self, dt, time_modelling):
        global current_time
        global time_processed_request
        global p_teor
        time_processed_request.clear()
        current_time = 0
        generator = self._generator
        processor = self._processor
        queue_size = [0]
        time_generated_request = []
        num_requests = [0]

        gen_period = generator.next_time_period()
        proc_period = gen_period + processor.next_time_period()

        while current_time < time_modelling:
            num = num_requests[-1]
            if gen_period <= current_time:
                time_generated_request.append(current_time)
                generator.emit_request()
                num += 1
                gen_period += generator.next_time_period()
            if proc_period <= current_time:
                if processor.current_queue_size > 0:
                    num -= 1
                processor.process()

                if processor.current_queue_size > 0:
                    proc_period += processor.next_time_period()
                else:
                    proc_period = gen_period + processor.next_time_period()
            queue_size.append(processor.current_queue_size)

            current_time += dt
            num_requests.append(num)

        lambda_fact = 1 / (sum(generator.time_periods) / len(generator.time_periods))
        mu_fact = 1 / (sum(processor.time_periods) / len(processor.time_periods))
        p = lambda_fact / mu_fact
        num_reports_teor = p / (1 - p)
        num_reports_fact = sum(queue_size) / len(queue_size)
        k = num_reports_fact / num_reports_teor

        if p_teor >= 1 or p_teor <= 0 or k == 0:
            k = 1

        if len(time_processed_request):
            mas_time_request_in_smo = []
            for i in range(len(time_processed_request)):
                mas_time_request_in_smo.append(time_processed_request[i] - time_generated_request[i])
            avg_time_in_smo = sum(mas_time_request_in_smo) / len(mas_time_request_in_smo) / k
        else:
            avg_time_in_smo = 0

        result = [
            processor.processed_requests,
            processor.reentered_requests,
            processor.max_queue_size,
            current_time,
            sum(queue_size) / len(queue_size),
            lambda_fact,
            mu_fact,
            avg_time_in_smo
        ]
        return result


def create_graph():
    intens = 0.001
    mas = []
    res = []
    while intens < 0.9:
        mas_i = []
        print(intens)
        for j in range(100):
            generator_intensity = intens
            generator_range = 1
            processor_intensity = 1
            processor_range = 1

            time_modelling = 10000
            step = 0.1

            generator = RequestGenerator(GaussDistribution(1 / generator_intensity, 1 / generator_range))

            k = (processor_intensity * processor_range) ** (-1.086)
            lambd = 1 / (processor_intensity * gamma(1 + 1 / k))
            processor = RequestProcessor(WeibullDistribution(k, lambd))

            model = Modeller(generator, processor)
            result = model.time_based_modelling(step, time_modelling)[7]    # среднее время пребывания в СМО

            mas_i.append(result)

        mas.append(intens)
        res.append(sum(mas_i) / len(mas_i))
        mas_i.clear()

        intens += 0.03

    plt.plot(mas, res, 'purple')
    plt.grid()
    plt.title('Среднее время нахождения в очереди')
    plt.ylabel('Среднее время пребывания заявки в очереди')
    plt.xlabel('Загрузка системы')
    plt.show()


p_teor = 0


def create_graph_button_clicked():
    create_graph()


class mywindow(QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_model.clicked.connect(self.modeling_button_clicked)
        self.ui.pushButton_graph.clicked.connect(create_graph_button_clicked)

    def modeling_button_clicked(self):
        try:
            global p_teor
            generator_intensity = self.ui.spinbox_intensivity_gen.value()
            generator_range = self.ui.spinbox_intensivity_gen_range.value()
            processor_intensity = self.ui.spinbox_intensivity_oa.value()
            processor_range = self.ui.spinbox_intensivity_oa_range.value()
            modelling_time = self.ui.spinbox_time_model.value()

            generator = RequestGenerator(GaussDistribution(1 / generator_intensity, 1 / generator_range))

            k = (processor_intensity * processor_range) ** (-1.086)
            lambd = 1 / (processor_intensity * gamma(1 + 1 / k))
            processor = RequestProcessor(WeibullDistribution(k, lambd))

            step = 0.1

            model = Modeller(generator, processor)
            result_tb = model.time_based_modelling(step, modelling_time)

            p = p_teor = generator_intensity / processor_intensity

            self.ui.teor_p.setText(str(round(p, 4)))
            self.ui.fact_p.setText(str(round(result_tb[5] / result_tb[6], 4)))

            self.ui.teor_time.setText(str(round(p / (1 - p) / generator_intensity, 4)))
            self.ui.fact_time.setText(str(round(result_tb[7], 4)))

            self.ui.quanity.setText(str(result_tb[0]))  # обработано заявок

        except Exception as e:
            error_msg = QMessageBox()
            error_msg.setText('Ошибка!\n' + repr(e))
            error_msg.show()
            error_msg.exec()


if __name__ == "__main__":
    app = QApplication([])
    application = mywindow()
    application.show()

    sys.exit(app.exec())