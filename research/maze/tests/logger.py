from common.events import Subscriber
from common.text_utils import TextTable


class TrainLogger(Subscriber):
    def __init__(self):
        self._subscribe('algorithm.train_episode', TrainLogger._on_train)
        self._subscribe('algorithm.eval', TrainLogger._on_eval)
        self._subscribe('train.summary', TrainLogger._on_summary)
        self._subscribe('timer', TrainLogger._on_timer)
        self._history = []
        self._record = {}
        self._table = TextTable([
            ['EPISODE'],
            ['REWARD', '%+.1f', 7],
            ['QMAX', '%+.1f', 7],
            ['FINAL_EVAL_STATE', '%s', 32],
            ['EVALUATION', '%+.1f'],
            ['TASK_DONE', '%s'],
            ['DURATION', '%.2f s'],
        ], vline=' '*3)

    # noinspection PyMethodMayBeStatic
    def _on_summary(self, info):
        if isinstance(info, (list, tuple)):
            for i in info:
                print(str(i))
        else:
            print(info)

    def _on_timer(self, info):
        self._record['DURATION'] = info['train']
        self._update_table()

    def _on_train(self, info):
        self._record['EPISODE'] = info['episode']
        self._record['REWARD'] = info['reward']
        self._record['QMAX'] = info['qmax']
        self._update_table()

    def _on_eval(self, info):
        self._record['EVALUATION'] = info['ave_reward']
        self._record['TASK_DONE'] = 'DONE' if info['ave_done'] == 1. else '%d%%' % (info['ave_done'] * 100)
        self._record['FINAL_EVAL_STATE'] = "[ %s ]" % ', '.join(['%2d' % c for c in info['final_state']])
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
        if len(self._table.records) == 1 or len(self._table.records) % 30 == 0:
            print("\n%s" % self._table.header)
        print(self._table.last_record)
