"""State and rolling averages for one tactile sensor grid."""

import numpy as np


class SensorDataBuffer:
    """Store calibrated sensor frames and their moving-window averages."""

    def __init__(self, n_row, n_col, window_size=3):
        self.n_row = int(n_row)
        self.n_col = int(n_col)
        self.windowSize = max(1, int(window_size))
        self.calData = np.zeros((self.n_row, self.n_col))
        self.rawData = np.zeros((self.n_row, self.n_col))
        self.diffData = np.zeros((self.n_row, self.n_col))
        self.diffPerData = np.zeros((self.n_row, self.n_col))
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))
        self.frame_sequence = 0

    def _recompute_averages(self):
        self.rawDataAve = np.flipud(np.mean(self.rawDataWin, axis=0))
        self.diffDataAve = np.flipud(np.mean(self.diffDataWin, axis=0))
        self.diffPerDataAve = np.flipud(np.mean(self.diffPerDataWin, axis=0))

    def setWindowSize(self, window_size):
        self.windowSize = max(1, int(window_size))
        self.rawDataWin = np.repeat(self.rawData[None, ...], self.windowSize, axis=0)
        self.diffDataWin = np.repeat(self.diffData[None, ...], self.windowSize, axis=0)
        self.diffPerDataWin = np.repeat(
            self.diffPerData[None, ...], self.windowSize, axis=0
        )
        self._recompute_averages()

    def getRaw(self, raw_data):
        self.rawData = raw_data

    def getCal(self, calibration_data):
        self.calData = calibration_data

    def calDiff(self):
        self.diffData = self.rawData - self.calData

    def calDiffPer(self):
        non_zero_mask = self.calData != 0
        self.diffPerData = np.zeros_like(self.calData, dtype=float)
        np.divide(
            100 * self.diffData,
            self.calData,
            out=self.diffPerData,
            where=non_zero_mask,
        )

    def clearData(self):
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))
        self.frame_sequence = 0

    def getWin(self, index):
        """Insert the current frame, preserving the legacy 1-based indexing."""
        normalized_index = (index - 1) % self.windowSize

        if index == self.windowSize:
            self.rawDataWin[:-1] = self.rawDataWin[1:]
            self.diffDataWin[:-1] = self.diffDataWin[1:]
            self.diffPerDataWin[:-1] = self.diffPerDataWin[1:]

        self.rawDataWin[normalized_index] = self.rawData
        self.diffDataWin[normalized_index] = self.diffData
        self.diffPerDataWin[normalized_index] = self.diffPerData

        if index >= self.windowSize:
            self._recompute_averages()
            self.frame_sequence += 1
