
class SmMarket:
    def __init__(self):
        self.name = ""
        self.product_dic = {}

    def add_category(self, product):
        self.product_dic[product.code] = product