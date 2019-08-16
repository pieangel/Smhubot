import os
import settings
from symbol import SmSymbol
from market import SmMarket
from product_category import SmProduct
from symbol_manager import SymbolManager


class SmMarketManager:
    def __init__(self):
        self.market_list = []
        self.product_dic = {}
        self.symmgr = SymbolManager()

    def add_market(self, name):
        market = self.find_market(name)
        if market is None:
            market = SmMarket()
            market.name = name
            self.market_list.append(market)
        return market

    def find_market(self, name):
        for market in self.market_list:
            if market.name == name:
                return market
        return None

    def add_product(self, product):
        if product is None:
            return
        self.product_dic[product.code] = product

    def find_product(self, product_code):
        for product in self.product_dic.items():
            if product[1].code == product_code:
                return product[1]
        return None

    def read_symbol_from_file(self):
        # 품목 코드를 로드한다.
        file_name = os.path.join(settings.BASE_DIR, 'mst/PMCODE.cod')
        line_len = 151
        with open(file_name, "rb") as f:
            while True:
                bytes = f.read(line_len)
                if not bytes:
                    break
                b_market = bytes[0:20]
                market_name = b_market.decode('cp949')
                b_exchange_code = bytes[24:29]
                exchange_code = b_exchange_code.decode('cp949')
                b_product_code = bytes[29:34]
                product_code = b_product_code.decode('cp949')
                b_product_name = bytes[37:87]
                prodcut_name = b_product_name.decode('cp949')
                b_product_name_kr = bytes[87:137]
                product_name_kr = b_product_name_kr.decode('cp949')
                market = self.add_market(market_name.strip())
                category = SmProduct()
                category.code = product_code.strip()
                category.exchange_code = exchange_code.strip()
                category.market_name = market_name.strip()
                category.name = prodcut_name.strip()
                category.name_kr = product_name_kr.strip()
                market.add_category(category)
                self.add_product(category)

        # 품목에 따른 심볼을 로드한다.
        file_name = os.path.join(settings.BASE_DIR, 'mst/JMCODE.cod')
        line_len = 491
        with open(file_name, "rb") as f:
            while True:
                bytes = f.read(line_len)
                if not bytes:
                    break
                b_symbol_code = bytes[0:32]
                symbol_code = b_symbol_code.decode('cp949')
                b_exchange_name = bytes[32:37]
                exchange_name = b_exchange_name.decode('cp949')
                b_product_index = bytes[37:41]
                product_index = b_product_index.decode('cp949')
                b_product_code = bytes[41:46]
                product_code = b_product_code.decode('cp949')
                b_decimal = bytes[51:56]
                decimal = b_decimal.decode('cp949')
                b_contract_size = bytes[61:81]
                contract_size = b_contract_size.decode('cp949')
                b_tick_size = bytes[81:101]
                tick_size = b_tick_size.decode('cp949')
                b_tick_value = bytes[101:121]
                tick_value = b_tick_value.decode('cp949')
                b_contract_seungsu = bytes[121:141]
                contract_seungsu = b_contract_seungsu.decode('cp949')
                b_display_digit = bytes[141:151]
                display_digit = b_display_digit.decode('cp949')
                b_name = bytes[151:183]
                symbol_name = b_name.decode('cp949')
                b_name_kr = bytes[183:215]
                symbol_name_kr = b_name_kr.decode('cp949')
                product = self.find_product(product_code.strip())
                if product is None:
                    continue
                symbol = SmSymbol()
                symbol.symbol_code = symbol_code.strip()
                symbol.index = int(product_index.strip())
                symbol.exchange_name = exchange_name.strip()
                symbol.product_code = product_code.strip()
                symbol.decimal = int(decimal.strip())
                symbol.contract_size = float(contract_size.strip())
                symbol.tick_size = float(tick_size)
                symbol.tick_value = float(tick_value)
                symbol.contract_seunsu = float(contract_seungsu)
                symbol.display_digit = display_digit
                symbol.name = symbol_name.strip()
                symbol.name_kr = symbol_name_kr.strip()
                symbol.market_name = product.market_name
                product.contract_seunsu = symbol.contract_seunsu
                product.add_symbol(symbol)
                self.symmgr.add_symbol(symbol)


