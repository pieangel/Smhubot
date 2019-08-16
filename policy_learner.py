import os
import locale
import logging
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer


logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code  # 종목코드
        # 차트 데이터
        self.chart_data = chart_data
        # 환경
        self.environment = Environment(stock_code, chart_data)  # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터 매칭
        self.training_data = training_data  # 학습 데이터
        # 샘플 초기 상태
        self.sample = None
        # 현재 학습 데이터 인덱스
        self.training_data_idx = -1
        # 입력은 환경에서 발생하는 모든 요소를 반영한다.
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        # 정첵 신경멍 객체 생성 및 매칭, 입력 계층 갯수, 출력 갯수, 학습률을 넣어 준다.
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()  # 가시화 모듈

    # 학습 초기화 - 다시 시작할 때 초기화 한다.
    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    # 학습을 수행하는 함수
    def fit(
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logger.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}".format(
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))
        # 에폭 요약 디렉토리를 만든다.
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        # 에포크에서 성공한  숫자
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            # 에포크 관련 정보 초기화
            loss = 0.
            # 반복 카운트
            itr_cnt = 0
            # 수익이 발생한 횟수 카운트
            win_cnt = 0
            # 탐험 횟수 - 심층 신경망이 아닌 탐색에 의한 회수
            exploration_cnt = 0
            # 배치 크기
            batch_size = 0
            # 긍정적인 학습 숫자
            pos_learning_cnt = 0
            # 부정적인 학습 숫자
            neg_learning_cnt = 0

            # 메모리 초기화
            memory_sample = []
            # 액션 리스트
            memory_action = []
            # 보상 리스트
            memory_reward = []
            # 확률 리스트
            memory_prob = []
            # 프로파일 가치
            memory_pv = []
            # 보유 주식수
            memory_num_stocks = []
            # 탐험 인덱스 기억
            memory_exp_idx = []
            # 학습 인덱스 기억
            memory_learning_idx = []
            
            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # 가시화 초기화
            self.visualizer.clear([0, len(self.chart_data)])

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                next_sample = self._build_sample()
                if next_sample is None:
                    break

                # 정책 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                memory_sample.append(next_sample)
                # 현재의 행동을 저장
                memory_action.append(action)
                # 직접 보상 저장
                memory_reward.append(immediate_reward)
                # 포트폴리오 가치 저장
                memory_pv.append(self.agent.portfolio_value)
                # 주식 갯수 저장
                memory_num_stocks.append(self.agent.num_remain)
                # 샘플값(입력값), 액션값, 보상값을 튜플로 만들어 저장
                memory = [(
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]
                ]
                if exploration:
                    # 탐색 인덱스를 저장한다.
                    memory_exp_idx.append(itr_cnt)
                    # 액션수만큼 갯수를 가진 리스트를 만들어 붙인다.
                    # 여기에는 출력계층(매수, 매도)에 대한 확률이 들어간다.
                    # 그러나 탐색일 때는 그 확률이 없으므로 이용할 수 없음 값으로 채운다.
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)
                else:
                    memory_prob.append(self.policy_network.prob)

                # 반복에 대한 정보 갱신
                batch_size += 1
                itr_cnt += 1
                # 탐험에 의한 것이면 탐험 갯수 추가
                exploration_cnt += 1 if exploration else 0
                # 지연보상이 발생했으면 승리한 회수 추가
                win_cnt += 1 if delayed_reward > 0 else 0


                # 지연보상이 없고 배치크기가 최대 메모리보다 크거나 같으면
                if delayed_reward == 0 and batch_size >= max_memory:
                    # 즉시 보상으로 지연보상 값을 대치하고
                    delayed_reward = immediate_reward
                    # 기준 포트폴리오 값도 바꾸어준다.
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if learning and delayed_reward != 0:
                    # 배치 학습 데이터 크기
                    batch_size = min(batch_size, max_memory)
                    # 배치 학습 데이터 생성
                    x, y = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)
                    if len(x) > 0:
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        # 정책 신경망 학습 시킴 - 지연 보상이 발생했을 때만 진행시킨다.
                        loss += self.policy_network.train_on_batch(x, y)
                        memory_learning_idx.append([itr_cnt, delayed_reward])
                    batch_size = 0

            # 에포크 관련 정보 가시화
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')

            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_remain,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            pos_learning_cnt, neg_learning_cnt, loss))

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 학습 관련 정보 로그 기록
        logger.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

    # 배치 학습 데이터를 가져옴
    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        # create an ndarray with all zeros
        x = np.zeros((batch_size, 1, self.num_features))
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)

        # 배열을 역순으로 도는 것은 가장 최근의 것에 가장 많은 점수를 주기 위해서이다.
        # 과거로 갈 수록 지연보상을 적용할 근거가 떨어지기 때문에 먼 과거의 일일 수록 할인 요인을 적용한다.
        for i, (sample, action, reward) in enumerate(
                reversed(memory[-batch_size:])):
            # x[i] 에는 특징값 수만큼 값들이 3차원으로 구성되어 들어갑니다.
            # 일단 2차원은 반드시 구성하고 데이터가 남을 경우 3차원을 구성하게 된다.
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            # y[i, action]에는 액션을 취하면 1.0, 취하지 않았으면 0.5로 들어갑니다.
            # action 0이면 매수이므로 [1.0, 0.5]라고 들어가게 됩니다.
            # action 1이면 매도이므로 [0.5, 1.0]라고 들어가게 됩니다.
            y[i, action] = (delayed_reward + 1) / 2
            # 할인요소를 i승하여 곱해줌.
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i
        return x, y

    # 샘플 생성함수
    def _build_sample(self):
        # 차트 데이터를 가져온다.
        self.environment.observe()
        # 학습데이터 크기 끝에 도달하지 않았다면
        if len(self.training_data) > self.training_data_idx + 1:
            # 학습 인덱스 증가
            self.training_data_idx += 1
            temp2 = self.training_data.iloc[self.training_data_idx]
            # 입력값 한세트를 가져옴
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            # 입력값 세트에 상태들을 더함 - 이것이 최종 입력 계층 값이 됨.
            self.sample.extend(self.agent.get_states())
            # 입력 계층 값을 반환한다.
            return self.sample
        return None

    # 모델을 로드해 거래를 한다.
    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        # 모델을 로드한다.
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)
