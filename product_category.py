class SmProduct:
    def __init__(self):
        self.code = ""
        self.name = ""
        self.name_kr = ""
        self.exchange = ""
        self.exchange_code = ""
        self.market_name = ""
        self.symbol_list = []
        self.contract_seunsu = 0.0

    def add_symbol(self, symbol):
        self.symbol_list.append(symbol)