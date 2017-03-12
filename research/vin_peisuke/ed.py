#!/usr/bin/env python3
from experiment.experiment_data import ExperimentData


def main():
    ed = ExperimentData(base_path='./data')
    print(ed.path('map/map_data.pkl'))


if __name__ == "__main__":
    main()
