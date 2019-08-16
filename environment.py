class Environment:

    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, symbol_code, chart_data=None):
        self.symbol_code = symbol_code
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1
        self.old_observation = None

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        # 이전 관찰값을 저장해 둔다.
        self.old_observation = self.observation
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    # 이전 값을 가져오는 함수
    def get_prev_price(self):
        if self.old_observation is not None:
            return self.old_observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
