from common.events import Subscriber
from common.text_utils import TextTable


class VinTrainLogger(Subscriber):
    def __init__(self):
        self._subscribe('algorithm.train', VinTrainLogger._on_train)
        self._subscribe('algorithm.eval', VinTrainLogger._on_eval)
        self._subscribe('train.summary', VinTrainLogger._on_summary)
        self._subscribe('timer', VinTrainLogger._on_timer)
        self._history = []
        self._record = {}
        self._table = TextTable([
            ['EPOCH'],
            ['TRAIN_ERR', '%.3f'],
            ['TRAIN_ACC', '%.1f%%'],
            ['EVAL_ACC', '%.1f%%'],
            ['DURATION', '%.2f s'],
        ], vline=' ' * 3)

    def _on_timer(self, info):
        self._record['DURATION'] = info['train']
        self._update_table()

    def _on_train(self, info):
        self._record['EPOCH'] = info['epoch']
        self._record['TRAIN_ERR'] = info['train_err']
        self._record['TRAIN_ACC'] = info['train_acc'] * 100
        self._update_table()

    def _on_eval(self, info):
        self._record['EVAL_ACC'] = info['acc'] * 100
        self._update_table()

    def _update_table(self):
        if self._record_ready():
            self._history.append(self._record)
            self._table.add_record(self._record)
            self._record = {}
            self._print_table()

    def _record_ready(self):
        for f in self._table.fields:
            if f not in self._record:
                return False
        return True

    def _print_table(self):
        if len(self._table.records) == 1 or (len(self._table.records) - 1) % 30 == 0:
            print("\n%s" % self._table.header)
        print(self._table.last_record)

    # noinspection PyMethodMayBeStatic
    def _on_summary(self, info):
        if isinstance(info, (list, tuple)):
            for i in info:
                print(str(i))
        else:
            print(info)
