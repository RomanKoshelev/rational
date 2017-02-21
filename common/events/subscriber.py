from common.events import EventSystem


class Subscriber(object):
    def __enter__(self):
        return self

    def __exit__(self):
        self._unsubcribe_all()

    def _subscribe(self, event: str, method: classmethod):
        EventSystem.subscribe(event, method, self)

    def _unsubcribe_all(self):
        EventSystem.unsubscribe(subscriber=self)
