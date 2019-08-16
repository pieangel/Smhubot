import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd

# 정체 신경망 객체
class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        # 입력 차원
        self.input_dim = input_dim
        # 학습률
        self.lr = lr

        # LSTM 신경망
        self.model = Sequential() 

        # 입력층
        self.model.add(LSTM(256, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.5))
        # 입력계층의 비선형화를 막기위한 배치정규화
        self.model.add(BatchNormalization())
        # 은닉충 추가
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        # 은닉층 배치정규화
        self.model.add(BatchNormalization())
        # 은닉층 추가
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        # 은닉층 배치 정규화
        self.model.add(BatchNormalization())
        # 출력층 추가
        self.model.add(Dense(output_dim))
        # 출력층 활성화 함수 추가
        self.model.add(Activation('sigmoid'))

        # 컴파일 - 최적화는 stochastic gradient descent, 손실함수는 최소제곱오류 사용
        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    # 모델 리셋
    def reset(self):
        self.prob = None

    # 샘플데이터에 대한 예상 - 확률을 반환한다.
    def predict(self, sample):
        #temp = np.array(sample)
        #temp2 = temp.reshape(1, -1, self.input_dim)
        #temp3 = self.model.predict(temp2)
        # 2차원 배열을 반납하므로 1차원 배열만 반납한다.
        # predict는 3차원 배열을 입력값으로 받는다.
        # 하나의 입력값일때는 3차원 형식을 잘 맞춰 줘야 한다.
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    # 배치 데이터를 가지고 학습을 한다. 입력과 출력 변수를 받는다.
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    # 모델을 저장한다.
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)
    # 모델을 로드한다.
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)