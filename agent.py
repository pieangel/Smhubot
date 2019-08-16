import numpy as np


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 4  # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0  # 거래세 미고려 (실제 0.3%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2,
        delayed_reward_threshold=.05):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # 이전 자산에서 이 임계치가 될 때까지는 보상을 지연시킨다.
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_remain = 0  # 보유 잔고 0 : 잔고 없음, 0보다 작으면 매도 잔고 0보다 크면 매수 잔고
        # 평가 손익 = 예탁금 + 현재 잔고에 따른 수익
        self.portfolio_value = 0  # balance + 현재 평가 손익
        # 바로 직전의 평가 손익
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        # 평가 손익
        self.open_profit_loss = 0.0
        # 잔고 평균가
        self.remain_average_price = 0.0
        # 제품 승수
        self.symbol_seunsu = 1000.0

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    # 에이전트 상태 초기화
    def reset(self):
        self.balance = self.initial_balance
        self.num_remain = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # 예탁잔고 설정
    def set_balance(self, balance):
        self.initial_balance = balance

    # 상태 가져오기
    def get_states(self):
        # 보유 잔고 - 보유 잔고를 참조하여 방향 결정에 활용 한다.
        self.ratio_hold = self.num_remain
        # 기본 자산(처음 주어진 자산)과 현재 자산의 비율 - 얼마의 수익이 나고 있는지 알 수 있다.
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    # 액션을 결정함
    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정 - 무작위 수가 엡실론 보다 작으면 임의의 행동을 결정한다.
        if np.random.rand() < epsilon:
            exploration = True
            # 결정할 수 있는 행동의 갯수에서 임의의 수를 가져온다. - 이것이 행동이 된다.
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        else:
            exploration = False
            # 샘플에 대한 심층 신경망의 확률을 구한다.
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            # 확률 중에서 가장 높은 것을 가져온다.
            action = np.argmax(probs)
            # 신뢰도를 저장한다. 0%에서 100%로 사이에 있다.
            confidence = probs[action]
        return action, confidence, exploration

    # 액션의 유효성을 결정한다.
    def validate_action(self, action):
        validity = True
        # 매수일 경우
        if action == Agent.ACTION_BUY:
            # 현재 잔고가 매수이고 매수 가능 최대 갯수에 도달했다면 더 이상 매수를 할 수 없다.
            if self.num_remain > 0 and self.num_remain >= self.max_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 현재 잔고가 매도이고 매도 가능 최대 갯수에 도달했다면 더 이상 매도를 할 수 없다.
            if self.num_remain < 0 and self.num_remain <= -self.max_trading_unit:
                validity = False
        return validity

    # 거래 최대 갯수를 결정한다.
    def decide_trading_unit(self, confidence):
        # 신뢰도가 없을 때는 최소 개수만
        if np.isnan(confidence):
            return self.min_trading_unit
        # 추가로 살 수 있는 갯수를 더한다.
        # 추가로 살 수 있는 갯수에 신뢰도를 곱한 값과 순수하게 추가로 살 수 있는 갯수 중에서 더 작은 수를 선택한다.
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    # 액션을 수행한다.
    def act(self, action, confidence):
        # 액션이 유효하지 않으면 아무일도 하지 않는다.
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0
        # 신뢰도를 가지고 매수할 단위를 판단
        trading_unit = self.decide_trading_unit(confidence)
        # 이전 잔고를 임시 저장
        old_remain = self.num_remain
        # 매수
        if action == Agent.ACTION_BUY:
            # 잔고가 있다면 일단 실현 손익을 먼저 계산한다.
            if self.num_remain < 0:
                # 실현 손익을 계산한다.
                trading_profit = self.num_remain * (curr_price - self.remain_average_price) * self.symbol_seunsu
                # 실현 손익을 잔고에 더한다.
                self.balance += trading_profit
                # 새로운 잔고를 설정한다.
                self.num_remain = trading_unit
                self.remain_average_price = curr_price
                self.open_profit_loss = 0
            elif self.num_remain > 0: # 잔고가 매수인 경우
                # 잔고에 더해 준다.
                self.num_remain += trading_unit
                # 평균가를 다시 계산해 준다.
                self.remain_average_price = (old_remain * self.remain_average_price + trading_unit * curr_price)\
                    / self.num_remain
                # 평가 손익을 다시 계산해 준다.
                self.open_profit_loss = self.num_remain * (curr_price - self.remain_average_price) \
                    * self.symbol_seunsu
            else:
                self.num_remain = trading_unit
                self.remain_average_price = curr_price
                self.open_profit_loss = 0.0
            self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 잔고가 있다면 일단 실현 손익을 먼저 계산한다.
            if self.num_remain > 0:
                # 실현 손익을 계산한다.
                trading_profit = self.num_remain * (curr_price - self.remain_average_price) * self.symbol_seunsu
                # 실현 손익을 잔고에 더한다.
                self.balance += trading_profit
                self.num_remain = -1 * trading_unit
                self.remain_average_price = curr_price
                self.open_profit_loss = 0.0
            elif self.num_remain < 0:  # 잔고가 매도인 경우
                # 잔고에 더해 준다.
                self.num_remain += -1 * trading_unit
                # 평균가를 다시 계산해 준다.
                self.remain_average_price = (old_remain * self.remain_average_price + trading_unit * curr_price) \
                    / self.num_remain
                # 평가 손익을 다시 계산해 준다.
                self.open_profit_loss = self.num_remain * (curr_price - self.remain_average_price)\
                    * self.symbol_seunsu
            else:
                self.num_remain = -1 * trading_unit
                self.remain_average_price = curr_price
                self.open_profit_loss = 0.0
            self.num_sell += 1  # 매도 횟수 증가
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가
            # 잔고가 있다면 평가 손익을 계산해 준다.
            self.open_profit_loss = self.num_remain * (curr_price - self.remain_average_price) * self.symbol_seunsu

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + self.open_profit_loss
        # 현재 수익을 계산
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # 즉시 보상 판단
        # 수익이 났으면 즉시 보상을 1을 준다.
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        # 현재 평가 손익 비율이 임계 수익 비율을 넘으면
        if profitloss > self.delayed_reward_threshold: # 임계 수익이 최대 수익률을 넘으면 보상
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold: # 감내할 수 있는 최대 손실이 발생한 경우 벌점
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else: # 수익이나 손실이 임계치에 도달하지 않았을 때 지연 보상은 없다.
            delayed_reward = 0
         # 즉시 보상과 지연보상을 반환한다.
        return self.immediate_reward, delayed_reward

