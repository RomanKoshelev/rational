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
            # ['TRAIN_FINAL_STATE', '%s', 18],
            # ['REWARD', '%+.1f', 7],
            ['TRAIN_Q_MAX', '%+.1f'],
            ['EVAL_FINAL_STATE_SAMPLE', '%s'],
            ['EVAL_REWARD', '%+.1f'],
            ['EVAL_TASK_DONE', '%-10s', None, TextTable.ALIGN_LEFT],
            ['DURATION', '%.2f s'],
        ], vline=' ' * 3)

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
        # self._record['REWARD'] = info['reward']
        self._record['TRAIN_Q_MAX'] = info['qmax']
        # self._record['TRAIN_FINAL_STATE'] = self._format_state(info['state'])
        self._update_table()

    def _on_eval(self, info):
        self._record['EVAL_REWARD'] = info['ave_reward']
        self._record['EVAL_TASK_DONE'] = '*' * int(10 * info['ave_done'])
        self._record['EVAL_FINAL_STATE_SAMPLE'] = self._format_state(info['state'])
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

    @staticmethod
    def _format_state(state):
        return "[%s]" % ','.join(['%+3d' % c if abs(c) > .5 else '  0' for c in state])
