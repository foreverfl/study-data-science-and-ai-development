"""
1. LSTM (Long Short-Term Memory)
- 적용 상황: 자연어 생성, 기계 번역, 음성 인식 등 다양한 시퀀스 문제에 사용됨.
- 장점: 기울기 소실 문제를 해결하여 긴 시퀀스를 더 잘 처리.
- 단점: 모델의 복잡성과 연산 비용이 높음.

2. GPT (Generative Pre-trained Transformer)
- 적용 상황: 텍스트 생성, 텍스트 요약, 질문 응답 등 다양한 자연어 처리 작업에 사용됨.
- 장점: 양방향 문맥을 활용하며, 미리 학습된 모델을 다양한 작업에 미세조정이 가능.
- 단점: 모델 크기가 크고 연산 비용이 높음.

3. Transformer
- 적용 상황: 기계 번역, 텍스트 분류, 텍스트 생성 등 다양한 자연어 처리 작업에 사용됨.
- 장점: 병렬 처리가 가능하여 빠른 학습과 예측이 가능하고, self-attention 메커니즘으로 문맥을 잘 파악.
- 단점: 모델 규모가 크며, 연산 비용이 높을 수 있음.

4. RNN (Recurrent Neural Networks)
- 적용 상황: 텍스트 생성, 시계열 분석, 음성 인식 등 시퀀스 데이터에 사용됨.
- 장점: 시퀀스의 순서 정보를 잘 반영할 수 있음.
- 단점: 긴 시퀀스에 대한 학습에서 기울기 소실 문제가 발생할 수 있음.

5. Markov Chain
- 적용 상황: 간단한 텍스트 생성, 음악 생성, 웹 페이지 순위 결정 등에 사용됨.
- 장점: 계산 비용이 낮고 간단한 모델로 빠르게 결과를 얻을 수 있음.
- 단점: 상태 전이가 메모리리스(memoryless)하여 더 복잡한 패턴을 학습하기 어려움.
"""
