class SymbolManager:

    def __init__(self):
        self.symbol_dic = {}

    def add_symbol(self, symbol):
        if symbol is None:
            return
        self.symbol_dic[symbol.symbol_code] = symbol

    def find_symbol(self, symbol_code):
        for symbol in self.symbol_dic.items():
            if symbol[1].symbol_code == symbol_code:
                return symbol[1]
        return None